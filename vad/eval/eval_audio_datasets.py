"""Audio-branch evaluation on public real-audio datasets.

  sesa  SESA (Freesound, CC-BY-4.0): casual / gunshot / explosion / siren clips.
        - CLAP zero-shot 4-way accuracy, (a) paper-style prompts "This is a sound of <label>"
          (MS-CLAP 2023 reports 0.65 zero-shot on SESA) and (b) our 7 audio classes collapsed to 4.
        - Audio Stage 1 trigger: per-clip max score ROC-AUC (threat vs casual), recall/FAR at τ_a.
        - Full two-stage audio branch: per-class recall, casual false-alarm rate, class accuracy.
  tut   TUT Rare Sound Events 2017 (DCASE 2017 Task 2) evaluation mixtures: 30 s everyday backgrounds,
        half with one gunshot / glass break / baby cry. Event-based onset-only metrics with a 500 ms
        collar (F1, ER=(FN+FP)/N, single class): Stage 1 only and Stage 1 + CLAP. Baby-cry mixtures
        are reported as a hard-negative false-trigger rate (no matching target in our system).
  rlvs  Real Life Violence Situations (Kaggle): 1,000 violent + 1,000 non-violent YouTube clips with
        sound. Video-level ROC-AUC/AP of video Stage 1, audio Stage 1 and fused scores; alert recall /
        false-alarm rate for video-only vs audio+video fusion. (Needs X3D-S extraction -> run on Colab.)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vad.audio.audio_io import load_audio
from vad.audio.clap_scorer import AUDIO_TO_VIDEO_CLASS, CLAP_SR, ClapScorer
from vad.audio.pipeline import audio_windows, fuse, run_audio
from vad.audio.trigger import DEFAULT_AUDIO_TAU, EAT_SR, AudioTrigger
from vad.eval.metrics import average_precision_score, roc_auc_score
from vad.stage2.clip_scorer import pick_device

SWEEP = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


def _rate(a, b):
    return round(a / b, 4) if b else 0.0


# ============================================================================ SESA
SESA_MAP = {"casual": None, "gunshot": "shooting", "explosion": "explosion", "siren": "fire_alarm"}
SESA_GROUP = {"gunshot": "shooting", "explosion": "explosion", "siren": "alarm"}


def eval_sesa(root: Path, out: Path, dev: str, tau_a: float) -> dict[str, Any]:
    files = sorted(root.rglob("*.wav"))
    trig, clap = AudioTrigger(device=dev), ClapScorer(device=dev)
    paper_labels = ["casual", "gunshot", "explosion", "siren"]
    paper = ClapScorer.__new__(ClapScorer)  # reuse loaded model with paper-style prompts
    paper.__dict__.update(clap.__dict__)
    paper.class_prompts = {c: [f"This is a sound of {c}"] for c in paper_labels}
    paper.classes = paper_labels
    paper.text_emb = clap._encode_prompts(paper.class_prompts)
    recs = []
    t0 = time.time()
    for f in files:
        lab = f.stem.split("_")[0]
        split = f.parent.name
        w32, w44 = load_audio(f, EAT_SR), load_audio(f, CLAP_SR)
        pp = paper.probs(w44)
        p7 = clap.score_segment(w44, CLAP_SR)
        mp = p7["mean_probs"]
        p4 = {"casual": mp["normal"] + mp["unrelated"] + mp["fighting"] + mp["glass_break"],
              "gunshot": mp["shooting"], "explosion": mp["explosion"], "siren": mp["fire_alarm"]}
        tr = trig.score(w32)
        ar = run_audio(w32, trig, clap, tau_a=tau_a, wav44k=w44)
        recs.append({
            "file": f.name, "split": split, "label": lab,
            "paper_pred": paper_labels[int(np.argmax(pp))],
            "ours4_pred": max(p4, key=p4.get), "ours7_pred": p7["class"], "ours7_suppressed": p7["suppressed"],
            "trig_max": float(tr.scores.max()), "trig_groups": {g: float(v.max()) for g, v in tr.group_scores.items()},
            "alert_classes": [e["class"] for e in ar["events"]], "n_windows": len(ar["windows"]),
        })
    res: dict[str, Any] = {"n_files": len(recs), "elapsed_sec": round(time.time() - t0, 1)}
    for split in ("test", "all"):
        R = [r for r in recs if split == "all" or r["split"] == "test"]
        y = np.array([r["label"] != "casual" for r in R], int)
        s = np.array([r["trig_max"] for r in R])
        per = {}
        for lab in paper_labels:
            L = [r for r in R if r["label"] == lab]
            want = SESA_MAP[lab]
            per[lab] = {
                "n": len(L),
                "paper_zero_shot_acc": _rate(sum(r["paper_pred"] == lab for r in L), len(L)),
                "ours4_acc": _rate(sum(r["ours4_pred"] == lab for r in L), len(L)),
                "two_stage_alert_rate": _rate(sum(bool(r["alert_classes"]) for r in L), len(L)),
                "two_stage_correct_class": _rate(sum(want in r["alert_classes"] for r in L), len(L)) if want else None,
            }
        res[split] = {
            "paper_zero_shot_acc": _rate(sum(r["paper_pred"] == r["label"] for r in R), len(R)),
            "ours4_acc": _rate(sum(r["ours4_pred"] == r["label"] for r in R), len(R)),
            "trigger_auc_threat_vs_casual": round(roc_auc_score(y, s), 4),
            "trigger_ap": round(average_precision_score(y, s), 4),
            "sweep": [{"tau_a": t, "threat_trigger_recall": _rate(int(((s > t) & (y == 1)).sum()), int(y.sum())),
                       "casual_trigger_rate": _rate(int(((s > t) & (y == 0)).sum()), int((y == 0).sum()))} for t in SWEEP],
            "two_stage_threat_recall": _rate(sum(bool(r["alert_classes"]) for r in R if r["label"] != "casual"), int(y.sum())),
            "two_stage_casual_far": _rate(sum(bool(r["alert_classes"]) for r in R if r["label"] == "casual"), int((y == 0).sum())),
            "per_class": per,
        }
    res["records"] = recs
    return res


def calibrate_sesa(root: Path, dev: str, max_far: float = 0.05) -> dict[str, Any]:
    """Choose τ_a on the SESA *train* split (never test): lowest τ_a whose full two-stage branch keeps the
    casual-sound false-alarm rate <= max_far; then report the test split at that τ_a."""
    trig, clap = AudioTrigger(device=dev), ClapScorer(device=dev)
    data = []
    for f in sorted(root.rglob("*.wav")):
        w32, w44 = load_audio(f, EAT_SR), load_audio(f, CLAP_SR)
        data.append({"split": f.parent.name, "label": f.stem.split("_")[0], "tr": trig.score(w32), "w44": w44, "cache": {}})

    def two_stage(d, t):
        alerts = []
        for w in audio_windows(d["tr"], t):
            key = (w["start_sec"], w["end_sec"])
            if key not in d["cache"]:
                d["cache"][key] = clap.score_segment(d["w44"][int(key[0] * CLAP_SR): int(key[1] * CLAP_SR)], CLAP_SR)
            r = d["cache"][key]
            if not r["suppressed"] and AUDIO_TO_VIDEO_CLASS.get(r["class"]):
                alerts.append(r["class"])
        return alerts

    grid = sorted({round(float(x), 2) for x in np.arange(0.02, 0.61, 0.02)} | {0.25})
    table = {}
    for split in ("train", "test"):
        D = [d for d in data if d["split"] == split]
        rows = []
        for t in grid:
            al = [two_stage(d, t) for d in D]
            thr = [i for i, d in enumerate(D) if d["label"] != "casual"]
            cas = [i for i, d in enumerate(D) if d["label"] == "casual"]
            want = {i: SESA_MAP[D[i]["label"]] for i in thr}
            rows.append({"tau_a": t, "threat_recall": _rate(sum(bool(al[i]) for i in thr), len(thr)),
                         "threat_correct_class": _rate(sum(want[i] in al[i] for i in thr), len(thr)),
                         "casual_far": _rate(sum(bool(al[i]) for i in cas), len(cas))})
        table[split] = rows
    ok = [r for r in table["train"] if r["casual_far"] <= max_far]
    chosen = min(ok, key=lambda r: r["tau_a"])["tau_a"] if ok else max(grid)
    test_at = next(r for r in table["test"] if r["tau_a"] == chosen)
    train_at = next(r for r in table["train"] if r["tau_a"] == chosen)
    return {"rule": f"lowest tau_a with train casual FAR <= {max_far}", "chosen_tau_a": chosen,
            "train_at_chosen": train_at, "test_at_chosen": test_at,
            "test_at_default_0.25": next(r for r in table["test"] if r["tau_a"] == 0.25),
            "table": table}


# ============================================================================ TUT
TUT_GROUP = {"gunshot": "shooting", "glassbreak": "glass_break", "babycry": None}
TUT_CLAP = {"gunshot": "shooting", "glassbreak": "glass_break"}


def _tut_annotations(root: Path) -> dict[str, tuple[str, float | None]]:
    """mixture filename -> (class, onset or None) from every event_list*.csv found under root."""
    ann: dict[str, tuple[str, float | None]] = {}
    for f in root.rglob("event_list*"):
        cls = next((c for c in TUT_GROUP if c in f.name), None)
        if cls is None or f.suffix not in (".csv", ".txt"):
            continue
        for line in f.read_text().splitlines():
            parts = [p for p in re.split(r"[\t,]", line.strip()) if p]
            if not parts or not parts[0].endswith(".wav"):
                continue
            onset = float(parts[1]) if len(parts) >= 3 else None
            ann[Path(parts[0]).name] = (cls, onset)
    return ann


def eval_tut(root: Path, out: Path, dev: str, tau_a: float, collar: float = 0.5, hop: float = 0.25) -> dict[str, Any]:
    ann = _tut_annotations(root)
    wavs = {p.name: p for p in root.rglob("*.wav") if p.name in ann}
    trig, clap = AudioTrigger(device=dev), ClapScorer(device=dev)
    rows = []
    t0 = time.time()
    for i, (name, (cls, onset)) in enumerate(sorted(ann.items())):
        p = wavs.get(name)
        if p is None:
            continue
        w32 = load_audio(p, EAT_SR)
        tr = trig.score(w32, window_sec=1.0, hop_sec=hop)
        w44 = load_audio(p, CLAP_SR)
        row = {"file": name, "class": cls, "onset": onset, "groups": {}}
        for g, v in tr.group_scores.items():
            v = np.where(tr.silent, 0.0, v)
            row["groups"][g] = v.astype(np.float32)
        # CLAP verdict around each candidate onset is computed lazily per threshold below
        row["_w44"] = w44
        row["_times"] = tr.times
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"[tut] {i + 1}/{len(ann)} mixtures ({time.time() - t0:.0f}s)", flush=True)

    def detect(row, group, t, use_clap):
        v = row["groups"][group]
        idx = np.flatnonzero(v > t)
        if idx.size == 0:
            return None
        est = float(row["_times"][idx[0]] + 0.5)  # centre of the first flagged 1 s window
        if use_clap:
            s0 = max(0.0, est - 1.0)
            seg = row["_w44"][int(s0 * CLAP_SR): int((est + 2.0) * CLAP_SR)]
            r = clap.score_segment(seg, CLAP_SR)
            if r["suppressed"] or r["class"] != TUT_CLAP[row["_target"]]:
                return None
        return est

    res: dict[str, Any] = {"n_mixtures": len(rows), "collar_sec": collar, "hop_sec": hop}
    for target in ("gunshot", "glassbreak"):
        R = [r for r in rows if r["class"] == target]
        for r in R:
            r["_target"] = target
        out_t = {}
        for mode in ("stage1", "stage1+clap"):
            sweep = []
            for t in SWEEP if mode == "stage1" else (tau_a,):
                tp = fp = fn = 0
                for r in R:
                    est = detect(r, TUT_GROUP[target], t, mode != "stage1")
                    ref = r["onset"]
                    if ref is None:
                        fp += int(est is not None)
                    elif est is None:
                        fn += 1
                    elif abs(est - ref) <= collar:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                n_ref = sum(r["onset"] is not None for r in R)
                sweep.append({"tau_a": t, "tp": tp, "fp": fp, "fn": fn, "n_ref": n_ref,
                              "f1": _rate(2 * tp, 2 * tp + fp + fn), "er": _rate(fn + fp, n_ref)})
            out_t[mode] = sweep
        res[target] = out_t
    # baby-cry mixtures: hard negatives (any trigger group firing)
    B = [r for r in rows if r["class"] == "babycry"]
    res["babycry_false_trigger"] = [{"tau_a": t, "rate": _rate(sum(any((v > t).any() for v in r["groups"].values()) for r in B), len(B)),
                                     "screaming_group_rate": _rate(sum((r["groups"]["screaming"] > t).any() for r in B), len(B))}
                                    for t in SWEEP]
    return res


# ============================================================================ RLVS
def eval_rlvs(root: Path, out: Path, dev: str, tau_a: float, checkpoint: Path, feat_dir: Path, limit: int) -> dict[str, Any]:
    from scripts.extract_x3d import embed_video, load_x3d
    from vad.pipeline.run_video import load_stage1_checkpoint, run_video
    from vad.stage2.clip_scorer import ClipScorer

    vids = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")])
    lab_of = lambda p: 0 if re.search(r"non[-_ ]?violence|^NV_", str(p), re.I) else 1  # noqa: E731
    if limit:
        v1 = [p for p in vids if lab_of(p) == 1][:limit]
        v0 = [p for p in vids if lab_of(p) == 0][:limit]
        vids = v1 + v0
    mil, ckpt = load_stage1_checkpoint(checkpoint, device=dev)
    tau_v, sw = float(ckpt["tau_info"]["tau"]), int(ckpt.get("smooth_window", 3))
    x3d, dtype = load_x3d(torch.device(dev))
    clip, trig, clap = ClipScorer(device=dev), AudioTrigger(device=dev), ClapScorer(device=dev)
    feat_dir.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for i, p in enumerate(vids):
        y = lab_of(p)
        npy = feat_dir / f"{p.stem}.npy"
        try:
            if not npy.exists():
                feats, n_raw, fps, _ = embed_video(x3d, dtype, torch.device(dev), str(p), batch=32)
                np.save(npy, feats)
                npy.with_suffix(".json").write_text(json.dumps({"fps": fps, "n_frames": n_raw, "stride": 6, "clip_len": 13}))
            vres = run_video(p, stage1_model=mil, scorer=clip, tau=tau_v, device=dev, features_npy=npy,
                             smooth_window=sw, include_suppressed=True)
            from vad.eval.metrics import score_clips
            from vad.data.ucf_dataset import l2_normalise
            _, sm = score_clips(mil, l2_normalise(np.load(npy)), device=dev, smooth_window=sw)
            ares = run_audio(str(p), trig, clap, tau_a=tau_a)
            fused = fuse(vres["events"], vres.get("suppressed_events", []), ares["events"])
            recs.append({"file": p.name, "label": y, "has_audio": ares["has_audio"], "silent_frac": ares["silent_frac"],
                         "v_score": float(sm.max()), "a_score": float(max(ares["scores"])) if ares["scores"] else 0.0,
                         "v_alert": bool(vres["events"]), "a_alert": bool(ares["events"]), "f_alert": bool(fused),
                         "v_classes": [e["class"] for e in vres["events"]], "f_classes": [a["class"] for a in fused],
                         "f_sources": [a["source"] for a in fused]})
        except Exception as exc:  # corrupt / undecodable clip
            recs.append({"file": p.name, "label": y, "error": f"{type(exc).__name__}: {exc}"})
        if (i + 1) % 100 == 0:
            print(f"[rlvs] {i + 1}/{len(vids)} videos ({time.time() - t0:.0f}s)", flush=True)
    ok = [r for r in recs if "error" not in r]
    y = np.array([r["label"] for r in ok])
    sv, sa = np.array([r["v_score"] for r in ok]), np.array([r["a_score"] for r in ok])
    res: dict[str, Any] = {"n_videos": len(recs), "n_ok": len(ok), "n_errors": len(recs) - len(ok),
                           "has_audio_rate": _rate(sum(r["has_audio"] for r in ok), len(ok)),
                           "mean_silent_frac": round(float(np.mean([r["silent_frac"] for r in ok])), 4) if ok else None,
                           "tau_v": tau_v, "tau_a": tau_a, "elapsed_sec": round(time.time() - t0, 1)}
    res["video_level"] = {n: {"roc_auc": round(roc_auc_score(y, s), 4), "ap": round(average_precision_score(y, s), 4)}
                          for n, s in [("video_stage1", sv), ("audio_stage1", sa), ("fused_mean", 0.5 * sv + 0.5 * sa),
                                       ("fused_max", np.maximum(sv, sa))]}
    for k, name in (("v_alert", "video_only"), ("a_alert", "audio_only"), ("f_alert", "fused")):
        res[name] = {"recall": _rate(sum(r[k] for r in ok if r["label"] == 1), int((y == 1).sum())),
                     "far": _rate(sum(r[k] for r in ok if r["label"] == 0), int((y == 0).sum()))}
    viol = [r for r in ok if r["label"] == 1]
    res["violent_alert_class_counts"] = {"video": dict(Counter(c for r in viol for c in r["v_classes"])),
                                         "fused": dict(Counter(c for r in viol for c in r["f_classes"])),
                                         "fused_sources": dict(Counter(s for r in ok for s in r["f_sources"]))}
    res["records"] = recs
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio-branch evaluation on SESA / TUT Rare 2017 / RLVS")
    ap.add_argument("dataset", choices=["sesa", "sesa-calibrate", "tut", "rlvs"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-a", type=float, default=DEFAULT_AUDIO_TAU)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint", default="checkpoints/best_seed416.pt")
    ap.add_argument("--feat-dir", default="/content/rlvs_feats")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    if a.dataset == "sesa":
        res = eval_sesa(root, out, dev, a.tau_a)
    elif a.dataset == "sesa-calibrate":
        res = calibrate_sesa(root, dev)
    elif a.dataset == "tut":
        res = eval_tut(root, out, dev, a.tau_a)
    else:
        res = eval_rlvs(root, out, dev, a.tau_a, Path(a.checkpoint), Path(a.feat_dir), a.limit)
    (out / f"{a.dataset}_eval.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    brief = {k: v for k, v in res.items() if k not in ("records",)}
    print(json.dumps(brief, indent=1, default=float)[:6000], flush=True)


if __name__ == "__main__":
    main()
