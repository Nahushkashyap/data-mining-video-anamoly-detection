"""Train + evaluate the home-security audio head on FSD50K (run on Colab; see colab/job_audio_home.py).

Stages (python -m vad.audio.train_home_head <stage> ...):
  select    FSD50K label CSVs -> manifest of target-class positives + household negatives (seed 415)
  download  fetch only the selected clips from the HF mirror Fhrozen/FSD50k (per-file, threaded, resumable)
  extract   1 s windows (hop 0.5 s, <=40 per clip, tiled to 10 s exactly as the app does) -> frozen EfficientAT
            mn10 embeddings [960] + the zero-shot trigger's scores (baseline)
  train     MIL head (max over a clip's windows), 3 seeds; per-class thresholds on FSD50K val;
            evaluation on the FSD50K eval split (uploader-disjoint from dev) vs the zero-shot trigger
  clap      optional: full two-stage numbers (trigger -> MS-CLAP veto) for old and new configurations

Labels in FSD50K include all AudioSet ancestors, so classes are defined with exclusions (see class_vector).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from vad.audio.home_head import GATE, HOME_CLASSES

HF_BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main"
EXCLUDE_FROM_NEG = {"Crack"}  # ambiguous impact sound: neither a positive nor a negative
PHONE_LIKE = {"Telephone", "Ringtone", "Doorbell"}  # carry the AudioSet 'Alarm' ancestor but are not alarms
HARD_NEGATIVES = [
    "Fireworks", "Knock", "Door", "Sliding_door", "Cupboard_open_or_close", "Drawer_open_or_close", "Chink_and_clink",
    "Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Walk_and_footsteps", "Speech", "Conversation", "Laughter",
    "Crying_and_sobbing", "Child_speech_and_kid_speaking", "Music", "Drum", "Bass_drum", "Snare_drum", "Crash_cymbal",
    "Dog", "Bark", "Cat", "Doorbell", "Telephone", "Ringtone", "Applause", "Clapping", "Cheering", "Hammer", "Drill",
    "Power_tool", "Typing", "Computer_keyboard", "Keys_jangling", "Coin_(dropping)", "Squeak", "Cough", "Sneeze",
    "Thunder", "Rain", "Wind", "Water_tap_and_faucet", "Toilet_flush", "Microwave_oven", "Vehicle",
    "Traffic_noise_and_roadway_noise", "Car_passing_by", "Vehicle_horn_and_car_horn_and_honking", "Engine",
    "Tearing", "Crumpling_and_crinkling", "Packing_tape_and_duct_tape", "Zipper_(clothing)", "Whoosh_and_swoosh_and_swish",
]
WIN = 32000  # 1 s at 32 kHz
HOP = 16000
MAX_WIN = 40
SILENCE_DBFS = -60.0


def class_vector(labels: set[str]) -> list[int]:
    gun = "Gunshot_and_gunfire" in labels
    return [
        int(bool(labels & {"Screaming", "Shout", "Yell"})),                              # scream
        int("Shatter" in labels),                                                       # glass_break
        int(bool(labels & {"Slam", "Thump_and_thud", "Boom"})),                          # loud_bang
        int(gun),                                                                       # shooting
        int("Explosion" in labels and not gun and "Fireworks" not in labels),           # explosion
        int("Siren" in labels or ("Alarm" in labels and not (labels & PHONE_LIKE))),    # alarm
    ]


# ----------------------------------------------------------------------------- select
def select(labels_dir: Path, out: Path, seed: int = 415, neg_dev: int = 4500, neg_eval: int = 2500,
           alarm_cap: int = 700, cap_hard_dev: int = 120, cap_hard_eval: int = 70, limit: int | None = None) -> None:
    rng = random.Random(seed)
    items = []
    for fn, default_split in (("dev.csv", None), ("eval.csv", "eval")):
        for r in csv.DictReader(open(labels_dir / fn, encoding="utf-8")):
            labels = set(r["labels"].split(","))
            items.append({"fname": r["fname"], "split": default_split or r["split"], "labels": sorted(labels),
                          "y": class_vector(labels)})
    chosen = []
    for part, neg_total, cap_hard in (("dev", neg_dev, cap_hard_dev), ("eval", neg_eval, cap_hard_eval)):
        pool = [it for it in items if (it["split"] == "eval") == (part == "eval")]
        pos = [it for it in pool if any(it["y"])]
        alarm_only = [it for it in pos if it["y"] == [0, 0, 0, 0, 0, 1]]
        if part == "dev" and len(alarm_only) > alarm_cap:
            drop = set(id(x) for x in rng.sample(alarm_only, len(alarm_only) - alarm_cap))
            pos = [it for it in pos if id(it) not in drop]
        neg_pool = [it for it in pool if not any(it["y"]) and not (set(it["labels"]) & EXCLUDE_FROM_NEG)]
        rng.shuffle(neg_pool)
        picked, seen = [], set()
        for lab in HARD_NEGATIVES:
            cands = [it for it in neg_pool if lab in it["labels"] and it["fname"] not in seen][:cap_hard]
            picked += cands
            seen.update(c["fname"] for c in cands)
        rest = [it for it in neg_pool if it["fname"] not in seen]
        picked += rest[: max(0, neg_total - len(picked))]
        chosen += pos + picked[:neg_total]
    if limit:
        rng.shuffle(chosen)
        chosen = chosen[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"classes": HOME_CLASSES, "seed": seed, "items": chosen}, open(out, "w"))
    _summary(chosen)


def _summary(items: list[dict]) -> None:
    for sp in ("train", "val", "eval"):
        sub = [it for it in items if it["split"] == sp]
        y = np.array([it["y"] for it in sub]) if sub else np.zeros((0, 6))
        pos = {c: int(y[:, k].sum()) for k, c in enumerate(HOME_CLASSES)}
        print(f"[select] {sp}: {len(sub)} clips, negatives {int((y.sum(1) == 0).sum())}, positives {pos}", flush=True)


# ----------------------------------------------------------------------------- download
def _clip_path(root: Path, it: dict) -> Path:
    return root / ("eval" if it["split"] == "eval" else "dev") / f"{it['fname']}.wav"


def download(manifest: Path, root: Path, token: str | None, workers: int = 32) -> None:
    import requests
    items = json.load(open(manifest))["items"]
    todo = [it for it in items if not (_clip_path(root, it).exists() and _clip_path(root, it).stat().st_size > 0)]
    print(f"[download] {len(items)} selected, {len(todo)} to fetch", flush=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(it: dict) -> str | None:
        p = _clip_path(root, it)
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_BASE}/clips/{'eval' if it['split'] == 'eval' else 'dev'}/{it['fname']}.wav"
        for attempt in range(8):
            try:
                r = requests.get(url, headers=headers, timeout=60)
                if r.status_code == 200 and r.content[:4] == b"RIFF":
                    tmp = p.with_suffix(".part")
                    tmp.write_bytes(r.content)
                    tmp.rename(p)
                    return None
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 2 ** attempt + random.random()))
                    continue
                return f"{it['fname']}: HTTP {r.status_code}"
            except Exception as exc:  # network hiccup
                time.sleep(min(60, 2 ** attempt))
                last = repr(exc)
        return f"{it['fname']}: gave up ({locals().get('last', 'rate limited')})"

    t0, done, fails = time.time(), 0, []
    with cf.ThreadPoolExecutor(workers) as ex:
        for err in ex.map(fetch, todo):
            done += 1
            if err:
                fails.append(err)
            if done % 500 == 0 or done == len(todo):
                print(f"[download] {done}/{len(todo)} ({time.time() - t0:.0f}s, {len(fails)} failed)", flush=True)
    if fails:
        print(f"[download] {len(fails)} failures, e.g. {fails[:5]}", flush=True)


# ----------------------------------------------------------------------------- extract
def _load32k(path: Path):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 32000:
        x = AF.resample(torch.from_numpy(x), sr, 32000).numpy()
    return x.astype(np.float32)


def _windows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = max(1, int(np.ceil(max(0, len(x) - WIN) / HOP)) + 1)
    pad = np.zeros((W - 1) * HOP + WIN, np.float32)
    pad[: len(x)] = x[: len(pad)]
    idx = np.arange(W) if W <= MAX_WIN else np.unique(np.linspace(0, W - 1, MAX_WIN).round().astype(int))
    ch = np.stack([pad[i * HOP: i * HOP + WIN] for i in idx])
    rms = np.sqrt(np.mean(np.square(ch, dtype=np.float64), axis=1))
    return ch, (20 * np.log10(np.maximum(rms, 1e-6))).astype(np.float32)


def extract(manifest: Path, root: Path, out: Path, device: str = "cuda", batch: int = 96) -> None:
    from vad.audio.trigger import AudioTrigger
    trig = AudioTrigger(device=device, use_head=False)
    items = [it for it in json.load(open(manifest))["items"] if _clip_path(root, it).exists()]
    print(f"[extract] {len(items)} clips with audio on disk", flush=True)
    feats, zs, cur, dbfs, offsets, keep = [], [], [], [], [0], []
    cur_idx = sorted({i for ix in trig.group_idx.values() for i in ix})
    zs_idx = [trig.home_zs_idx[c] for c in HOME_CLASSES]
    t0 = time.time()

    def prep(it):
        try:
            return it, _windows(_load32k(_clip_path(root, it)))
        except Exception as exc:
            return it, exc

    buf_ch, buf_meta = [], []

    def flush():
        if not buf_ch:
            return
        ch = np.concatenate(buf_ch)
        p, f = trig.probs_and_features(ch, batch=batch)
        k = 0
        for it, n, d in buf_meta:
            pp = p[k:k + n]
            feats.append(f[k:k + n].astype(np.float16))
            zs.append(np.stack([pp[:, ix].max(1) if ix else np.zeros(n) for ix in zs_idx], 1).astype(np.float16))
            c = pp[:, cur_idx].max(1)
            c[d < SILENCE_DBFS] = 0.0
            cur.append(c.astype(np.float16))
            dbfs.append(d)
            offsets.append(offsets[-1] + n)
            keep.append(it)
            k += n
        buf_ch.clear()
        buf_meta.clear()

    def bounded(ex, chunk=64):
        # ex.map over everything would decode all clips ahead of the GPU and hold them in RAM (OOM-killed at
        # 11.6 GB on the 12 GB Colab VM, 22 Nov); decode at most `chunk` clips ahead instead.
        for k in range(0, len(items), chunk):
            yield from ex.map(prep, items[k:k + chunk])

    with cf.ThreadPoolExecutor(4) as ex:
        for i, (it, res) in enumerate(bounded(ex), 1):
            if isinstance(res, Exception):
                print(f"[extract] skip {it['fname']}: {res!r}", flush=True)
                continue
            ch, d = res
            buf_ch.append(ch)
            buf_meta.append((it, len(ch), d))
            if sum(len(c) for c in buf_ch) >= 1024:
                flush()
            if i % 1000 == 0:
                print(f"[extract] {i}/{len(items)} clips, {offsets[-1]} windows, {time.time() - t0:.0f}s", flush=True)
    flush()
    split_code = {"train": 0, "val": 1, "eval": 2}
    np.savez(out, feats=np.concatenate(feats), zs=np.concatenate(zs), cur=np.concatenate(cur), dbfs=np.concatenate(dbfs),
             offsets=np.array(offsets, np.int64), y=np.array([it["y"] for it in keep], np.int8),
             split=np.array([split_code[it["split"]] for it in keep], np.int8),
             fname=np.array([it["fname"] for it in keep]), labels=np.array([",".join(it["labels"]) for it in keep]))
    print(f"[extract] saved {len(keep)} clips / {offsets[-1]} windows -> {out} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- train / eval
def _bags(d, idx, mean, std, device):
    import torch
    off = d["offsets"]
    B = len(idx)
    X = np.zeros((B, MAX_WIN, d["feats"].shape[1]), np.float16)
    M = np.zeros((B, MAX_WIN), bool)
    for b, i in enumerate(idx):
        s, e = off[i], off[i + 1]
        X[b, : e - s] = ((d["feats"][s:e].astype(np.float32) - mean) / std).astype(np.float16)
        m = d["dbfs"][s:e] >= SILENCE_DBFS
        if not m.any():
            m[np.argmax(d["dbfs"][s:e])] = True
        M[b, : e - s] = m
    return torch.from_numpy(X).to(device), torch.from_numpy(M).to(device)


def _clip_max(d, idx, key):
    off = d["offsets"]
    out = []
    for i in idx:
        s, e = off[i], off[i + 1]
        v = d[key][s:e].astype(np.float32)
        m = d["dbfs"][s:e] >= SILENCE_DBFS
        out.append(v[m].max(0) if m.any() else v.max(0))
    return np.array(out)


def _pool_logits(net, X, M):
    import torch
    logits = net(X.float())  # [B, W, C]
    return logits.masked_fill(~M.unsqueeze(-1), -1e4).max(dim=1).values


def _predict(net, X, M, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(_pool_logits(net, X[i:i + bs], M[i:i + bs])).cpu().numpy())
    return np.concatenate(out)


def _choose_threshold(p, y, neg_mask, max_neg_far=0.02):
    """F1-optimal threshold on val, raised if it fires on more than `max_neg_far` of val negatives."""
    cands = np.unique(np.concatenate([p, np.linspace(0.05, 0.95, 19)]))
    best_t, best_f = 0.5, -1
    for t in cands:
        pr = p >= t
        tp, fp, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
        f = 2 * tp / max(1, 2 * tp + fp + fn)
        if f > best_f:
            best_f, best_t = f, float(t)
    negs = p[neg_mask]
    if len(negs) and (negs >= best_t).mean() > max_neg_far:
        best_t = float(np.quantile(negs, 1 - max_neg_far) + 1e-6)
    return best_t


def _metrics(p, Y, thr=None):
    from vad.eval.metrics import average_precision_score, roc_auc_score
    out = {}
    for k, c in enumerate(HOME_CLASSES):
        y = Y[:, k]
        ok = 0 < y.sum() < len(y)  # AP/AUC undefined without both positives and negatives
        r = {"n_pos": int(y.sum()), "ap": round(float(average_precision_score(y, p[:, k])), 4) if ok else float("nan"),
             "roc_auc": round(float(roc_auc_score(y, p[:, k])), 4) if ok else float("nan")}
        if thr is not None:
            pr = p[:, k] >= thr[k]
            tp, fp, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
            r.update(precision=round(tp / max(1, tp + fp), 4), recall=round(tp / max(1, tp + fn), 4),
                     f1=round(2 * tp / max(1, 2 * tp + fp + fn), 4))
        out[c] = r
    out["mAP"] = round(float(np.nanmean([out[c]["ap"] for c in HOME_CLASSES])), 4)
    out["mean_auc"] = round(float(np.nanmean([out[c]["roc_auc"] for c in HOME_CLASSES])), 4)
    return out


def _operating_point(fires, Y):
    """Deployed-system view: does the trigger fire at all on a clip? recall per class + false-alarm rate on negatives."""
    neg = Y.sum(1) == 0
    res = {c: round(float(fires[Y[:, k] == 1].mean()), 4) if (Y[:, k] == 1).any() else float("nan")
           for k, c in enumerate(HOME_CLASSES)}
    res["false_alarm_rate_on_negatives"] = round(float(fires[neg].mean()), 4)
    res["n_negatives"] = int(neg.sum())
    return res


def train(npz: Path, out_dir: Path, seeds=(415, 416, 417), epochs: int = 40, device: str = "cuda") -> None:
    import torch
    from vad.audio.home_head import HomeHeadNet, calibrate
    d = dict(np.load(npz, allow_pickle=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = d["split"]
    tr, va, ev = (np.where(sp == k)[0] for k in (0, 1, 2))
    off = d["offsets"]
    tr_rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in tr])
    F = d["feats"][tr_rows].astype(np.float32)
    mean, std = F.mean(0), F.std(0) + 1e-5
    del F
    Xtr, Mtr = _bags(d, tr, mean, std, device)
    Xva, Mva = _bags(d, va, mean, std, device)
    Xev, Mev = _bags(d, ev, mean, std, device)
    Y = d["y"].astype(np.float32)
    Ytr = torch.from_numpy(Y[tr]).to(device)
    pos = Y[tr].sum(0)
    pos_w = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1), 1, 15), dtype=torch.float32, device=device)
    print(f"[train] train {len(tr)} / val {len(va)} / eval {len(ev)} clips; train positives {dict(zip(HOME_CLASSES, pos.astype(int).tolist()))}", flush=True)

    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = HomeHeadNet(Xtr.shape[-1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best = (-1.0, None, -1)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(tr), device=device)
            for i in range(0, len(tr), 64):
                b = perm[i:i + 64]
                M = Mtr[b] & (torch.rand(Mtr[b].shape, device=device) < 0.8)  # window dropout
                M[~M.any(1)] = Mtr[b][~M.any(1)]
                loss = lossf(_pool_logits(net, Xtr[b], M), Ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            vm = _metrics(_predict(net, Xva, Mva), Y[va])["mAP"]
            if best[1] is None or vm > best[0]:
                best = (vm, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
        net.load_state_dict(best[1])
        net.eval()
        pva, pev = _predict(net, Xva, Mva), _predict(net, Xev, Mev)
        neg_va = Y[va].sum(1) == 0
        thr = np.array([_choose_threshold(pva[:, k], Y[va][:, k], neg_va) for k in range(len(HOME_CLASSES))])
        ev_m = _metrics(pev, Y[ev], thr)
        fires = (calibrate(pev, thr) >= GATE).any(1)
        runs.append({"seed": seed, "best_epoch": best[2], "val_mAP": round(best[0], 4),
                     "thresholds": dict(zip(HOME_CLASSES, map(float, thr))), "eval": ev_m,
                     "eval_operating_point": _operating_point(fires, Y[ev]), "state_dict": best[1], "pev": pev})
        print(f"[train] seed {seed}: best epoch {best[2]}, val mAP {best[0]:.4f}, eval mAP {ev_m['mAP']:.4f}, "
              f"eval mean AUC {ev_m['mean_auc']:.4f}", flush=True)

    # zero-shot baselines on exactly the same eval clips
    zs_ev = _clip_max(d, ev, "zs")
    cur_ev = _clip_max(d, ev, "cur")
    base = {"zeroshot_class_scores": _metrics(zs_ev, Y[ev]),
            "current_trigger_at_0.25": _operating_point(cur_ev >= 0.25, Y[ev]),
            "note": "zeroshot_class_scores = AudioSet probabilities of the matching labels (loud_bang uses Bang/Slam/"
                    "Thump/Smash/Knock, which the deployed zero-shot trigger does not even listen for); "
                    "current_trigger_at_0.25 = the deployed zero-shot trigger (5 groups, tau_a 0.25)."}
    best_run = max(runs, key=lambda r: r["val_mAP"])
    agg = {}
    for c in HOME_CLASSES + ("mAP", "mean_auc"):
        if c in ("mAP", "mean_auc"):
            v = [r["eval"][c] for r in runs]
        else:
            v = [r["eval"][c]["ap"] for r in runs]
        agg[c] = {"mean": round(float(np.nanmean(v)), 4), "std": round(float(np.nanstd(v)), 4)}
    summary = {
        "trained": time.strftime("%Y-%m-%d %H:%M"), "classes": HOME_CLASSES,
        "n_clips": {"train": int(len(tr)), "val": int(len(va)), "eval": int(len(ev))},
        "eval_negatives": int((Y[ev].sum(1) == 0).sum()),
        "eval_positives": {c: int(Y[ev][:, k].sum()) for k, c in enumerate(HOME_CLASSES)},
        "head_eval_ap_mean_std_3_seeds": agg,
        "runs": [{k: v for k, v in r.items() if k not in ("state_dict", "pev")} for r in runs],
        "deployed_seed": best_run["seed"], "baseline": base,
    }
    json.dump(summary, open(out_dir / "home_head_eval.json", "w"), indent=1)
    torch.save({"version": "home_head_v1", "trained": summary["trained"], "seed": best_run["seed"],
                "val_map": best_run["val_mAP"], "classes": list(HOME_CLASSES), "in_dim": int(Xtr.shape[-1]),
                "hidden": 256, "feat_mean": mean, "feat_std": std, "thresholds": best_run["thresholds"],
                "state_dict": {k: v.cpu() for k, v in best_run["state_dict"].items()}}, out_dir / "home_head_v1.pt")
    np.savez(out_dir / "eval_scores.npz", head=best_run["pev"], zs=zs_ev, cur=cur_ev, y=Y[ev], fname=d["fname"][ev],
             labels=d["labels"][ev])
    print(json.dumps({"head_eval_ap": agg, "head_operating_point(deployed seed)": best_run["eval_operating_point"],
                      "baseline": {k: v for k, v in base.items() if k != "note"}}, indent=1), flush=True)


# ----------------------------------------------------------------------------- two-stage (with CLAP)
AUDIO_CLASS_TO_HOME = {"fighting": "scream", "glass_break": "glass_break", "loud_bang": "loud_bang",
                       "shooting": "shooting", "explosion": "explosion", "fire_alarm": "alarm"}


def clap_eval(out_dir: Path, root: Path, device: str = "cuda") -> None:
    """Old system (zero-shot trigger @0.25 -> CLAP v0 prompts) vs new (head @GATE -> CLAP v1 prompts), clip level."""
    from vad.audio import clap_scorer as cs
    from vad.audio.home_head import calibrate
    import torch
    s = np.load(out_dir / "eval_scores.npz", allow_pickle=False)
    ck = torch.load(out_dir / "home_head_v1.pt", map_location="cpu", weights_only=False)
    thr = np.array([ck["thresholds"][c] for c in HOME_CLASSES])
    Y = s["y"]
    new_fire = (calibrate(s["head"], thr) >= GATE).any(1)
    old_fire = s["cur"] >= 0.25
    res = {}
    for name, fire, prompts in (("old_zeroshot_trigger+clap_v0", old_fire, cs.AUDIO_CLASS_PROMPTS_V0),
                                ("new_home_head+clap_v1", new_fire, cs.AUDIO_CLASS_PROMPTS_V1)):
        scorer = cs.ClapScorer(device=device, class_prompts=prompts)
        alert = np.zeros(len(Y), bool)
        correct = np.zeros(len(Y), bool)
        for i in np.where(fire)[0]:
            wav = _load_any(_clip_path(root, {"fname": str(s["fname"][i]), "split": "eval"}), cs.CLAP_SR)
            p = scorer.probs(wav)
            classes = list(prompts)
            non = sum(p[k] for k, c in enumerate(classes) if c in ("normal", "unrelated"))
            if non >= cs.DEFAULT_SUPPRESS_THRESHOLD:
                continue
            alert[i] = True
            top = classes[int(np.argmax([p[k] if c not in ("normal", "unrelated") else -1 for k, c in enumerate(classes)]))]
            h = AUDIO_CLASS_TO_HOME.get(top)
            correct[i] = h is not None and Y[i][HOME_CLASSES.index(h)] == 1
        r = _operating_point(alert, Y)
        r["class_correct_given_alert"] = {c: round(float(correct[(Y[:, k] == 1) & alert].mean()) if ((Y[:, k] == 1) & alert).any() else 0.0, 4)
                                          for k, c in enumerate(HOME_CLASSES)}
        res[name] = r
        print(f"[clap] {name}: {json.dumps(r)}", flush=True)
        del scorer
        torch.cuda.empty_cache()
    json.dump(res, open(out_dir / "two_stage_eval.json", "w"), indent=1)


def _load_any(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    x, s0 = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return AF.resample(torch.from_numpy(x), s0, sr).numpy() if s0 != sr else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "download", "extract", "train", "clap"])
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--token-file", type=Path)
    a = ap.parse_args()
    if a.stage == "select":
        select(a.labels, a.manifest, limit=a.limit)
    elif a.stage == "download":
        tok = a.token_file.read_text().strip() if a.token_file and a.token_file.exists() else None
        download(a.manifest, a.root, tok)
    elif a.stage == "extract":
        extract(a.manifest, a.root, a.npz)
    elif a.stage == "train":
        train(a.npz, a.out, epochs=a.epochs)
    else:
        clap_eval(a.out, a.root)


if __name__ == "__main__":
    main()
