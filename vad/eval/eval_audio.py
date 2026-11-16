"""Evaluate the two-stage audio branch and audio+video fusion on the 226 UCF-Crime test videos.

Raw XD-Violence audio is not obtainable (official links need a Xidian login / Baidu account), so the
real-audio evaluation runs on the UCF test videos' own audio tracks. Reports:
  1. Silence audit per class (share of digitally silent 1 s windows; the "silence shortcut").
  2. Frame-level ROC-AUC / AP over all 226 test videos: video Stage 1, audio Stage 1 (trigger),
     mean fusion and max fusion.
  3. Audio Stage 2 (CLAP) on ground-truth anomaly windows: top-1 vs UCF class (via audio->video map).
  4. τ_a sweep: audio-only video-level recall / false-alarm rate.
  5. End-to-end at the main τ_a: video-only vs fused alerts (recall, normal-video FAR, per-video top-1).

Run on Colab (see colab/job_audio.py) or locally with the same arguments.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vad.audio.audio_io import SILENCE_DBFS, load_audio, rms_dbfs
from vad.audio.clap_scorer import AUDIO_CLASSES, AUDIO_TO_VIDEO_CLASS, CLAP_SR, ClapScorer
from vad.audio.pipeline import audio_windows, fuse
from vad.audio.trigger import DEFAULT_AUDIO_TAU, EAT_SR, AudioTrigger
from vad.data.ucf_dataset import build_frame_labels, l2_normalise, load_manifest, load_temporal_annotations
from vad.eval.metrics import average_precision_score, clips_to_frames, roc_auc_score, score_clips
from vad.pipeline.run_video import load_stage1_checkpoint, run_video
from vad.stage2.clip_scorer import UCF_TO_STAGE2_CLASS, ClipScorer, pick_device


def _rate(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifests/ucf_subset.csv")
    ap.add_argument("--annotations", default="data/splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-a", type=float, default=DEFAULT_AUDIO_TAU)
    ap.add_argument("--sweep", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    t_start = time.time()
    dev = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_manifest(args.manifest, lanes=("L1", "L2")) if r["split"] == "test"]
    if args.limit:
        rows = [r for r in rows if r["label"] == "1"][: args.limit] + [r for r in rows if r["label"] == "0"][: args.limit]
    ann = load_temporal_annotations(args.annotations)
    sweep = sorted({float(x) for x in args.sweep.split(",")} | {args.tau_a})

    mil, ckpt = load_stage1_checkpoint(args.checkpoint, device=dev)
    tau_v = float(ckpt["tau_info"]["tau"])
    sw = int(ckpt.get("smooth_window", 3))
    clip = ClipScorer(device=dev)
    trigger = AudioTrigger(device=dev)
    clap = ClapScorer(device=dev)
    print(f"[eval_audio] {len(rows)} videos | device={dev} | tau_v={tau_v:.4f} tau_a={args.tau_a}", flush=True)

    fl_lab, fl_v, fl_a = [], [], []
    silence = defaultdict(lambda: {"videos": 0, "no_audio": 0, "windows": 0, "silent_windows": 0, "all_silent_videos": 0})
    gt_recs: list[dict[str, Any]] = []
    sweep_counts = {t: {"anom_alert": 0, "norm_alert": 0, "anom_trig": 0, "norm_trig": 0, "flag_windows": 0, "windows": 0}
                    for t in sweep}
    e2e = {"video": Counter(), "fused": Counter(), "audio": Counter()}
    pervid = {"video": [], "fused": []}
    per_video_out: list[dict[str, Any]] = []
    timing = Counter()
    n_anom = n_norm = 0

    for i, r in enumerate(rows):
        vid, cls, label = r["video_id"], r["cls"], int(r["label"])
        vpath = Path(args.videos) / r["rel_path"]
        fpath = Path(args.features) / f"{vid}.npy"
        n_anom += label
        n_norm += 1 - label
        # ---------------- video branch (features already extracted) ----------------
        t0 = time.time()
        vres = run_video(vpath, stage1_model=mil, scorer=clip, tau=tau_v, device=dev, features_npy=fpath,
                         smooth_window=sw, include_suppressed=True)
        timing["video_sec"] += time.time() - t0
        feats = np.load(fpath)
        _, sm = score_clips(mil, l2_normalise(feats), device=dev, smooth_window=sw)
        n_frames, fps = vres["n_frames"], vres["fps"]
        fpc = 78  # X3D-S clip = 13 frames x stride 6
        labels = build_frame_labels(ann.get(vid, ("", ()))[1], n_frames) if label else np.zeros(n_frames, np.int32)
        fl_lab.append(labels)
        fl_v.append(clips_to_frames(sm, n_frames, fpc))

        # ---------------- audio branch ----------------
        t0 = time.time()
        wav32 = load_audio(vpath, EAT_SR)
        wav44 = load_audio(vpath, CLAP_SR) if wav32 is not None else None
        timing["decode_sec"] += time.time() - t0
        s = silence[cls]
        s["videos"] += 1
        audio_events: list[dict] = []
        if wav32 is None or len(wav32) == 0:
            s["no_audio"] += 1
            fl_a.append(np.zeros(n_frames, np.float32))
            trig = None
        else:
            t0 = time.time()
            trig = trigger.score(wav32)
            timing["trigger_sec"] += time.time() - t0
            timing["audio_sec"] += len(wav32) / EAT_SR
            s["windows"] += len(trig.scores)
            s["silent_windows"] += int(trig.silent.sum())
            s["all_silent_videos"] += int(trig.silent.all())
            sec = np.arange(n_frames) / fps
            fl_a.append(trig.scores[np.minimum((sec // 1.0).astype(int), len(trig.scores) - 1)])
            clap_cache: dict[tuple[float, float], dict] = {}

            def clap_on(w: dict) -> dict:
                key = (w["start_sec"], w["end_sec"])
                if key not in clap_cache:
                    tc = time.time()
                    seg = wav44[int(key[0] * CLAP_SR): int(key[1] * CLAP_SR)]
                    clap_cache[key] = clap.score_segment(seg, CLAP_SR)
                    timing["clap_sec"] += time.time() - tc
                    timing["clap_windows"] += 1
                return clap_cache[key]

            for t in sweep:
                wins = audio_windows(trig, t)
                sc = sweep_counts[t]
                sc["windows"] += len(trig.scores)
                sc["flag_windows"] += int((trig.scores > t).sum())
                kept = []
                for k, w in enumerate(wins):
                    res = clap_on(w)
                    if not res["suppressed"]:
                        kept.append({"id": k, **w, **res, "video_class": AUDIO_TO_VIDEO_CLASS.get(res["class"])})
                key = "anom" if label else "norm"
                sc[f"{key}_trig"] += int(bool(wins))
                sc[f"{key}_alert"] += int(bool(kept))
                if t == args.tau_a:
                    audio_events = kept

        # ---------------- audio Stage 2 on GT windows ----------------
        if label and wav44 is not None:
            for (s0, e0) in ann.get(vid, ("", ()))[1]:
                seg = wav44[int(max(0, s0 - 1) / fps * CLAP_SR): int(e0 / fps * CLAP_SR)]
                silent = rms_dbfs(seg) < SILENCE_DBFS
                rec = {"video_id": vid, "gt_class": UCF_TO_STAGE2_CLASS[cls], "interval": [s0, e0], "silent": bool(silent)}
                if not silent and len(seg) > 0:
                    res = clap.score_segment(seg, CLAP_SR)
                    rec.update(pred=res["class"], pred_video=AUDIO_TO_VIDEO_CLASS.get(res["class"]),
                               pred_anom=res["top3_anomaly"][0]["class"], suppressed=res["suppressed"], top3=res["top3"])
                gt_recs.append(rec)
        elif label:
            for (s0, e0) in ann.get(vid, ("", ()))[1]:
                gt_recs.append({"video_id": vid, "gt_class": UCF_TO_STAGE2_CLASS[cls], "interval": [s0, e0],
                                "silent": True, "no_audio": True})

        # ---------------- end-to-end alerts ----------------
        v_alerts = vres["events"]
        f_alerts = fuse(vres["events"], vres.get("suppressed_events", []), audio_events)
        key = "anom" if label else "norm"
        e2e["video"][key] += int(bool(v_alerts))
        e2e["fused"][key] += int(bool(f_alerts))
        e2e["audio"][key] += int(bool(audio_events))
        for src in ("video", "audio+video", "audio-confirmed", "audio"):
            e2e["fused"][f"{key}_{src}"] += sum(1 for a in f_alerts if a["source"] == src)
        if label:
            gt = UCF_TO_STAGE2_CLASS[cls]
            if v_alerts:
                pervid["video"].append((gt, max(v_alerts, key=lambda a: a["confidence"])["class"]))
            if f_alerts:
                pervid["fused"].append((gt, max(f_alerts, key=lambda a: a["confidence"])["class"]))
        per_video_out.append({
            "video_id": vid, "class": cls, "label": label,
            "has_audio": wav32 is not None and len(wav32) > 0,
            "silent_frac": round(float(trig.silent.mean()), 4) if trig is not None else 1.0,
            "max_trigger": round(float(trig.scores.max()), 4) if trig is not None and len(trig.scores) else 0.0,
            "video_alerts": [{k: a[k] for k in ("start_sec", "end_sec", "class", "confidence")} for a in v_alerts],
            "audio_alerts": [{k: a[k] for k in ("start_sec", "end_sec", "class", "confidence", "trigger_group")} for a in audio_events],
            "fused_alerts": [{k: a[k] for k in ("start_sec", "end_sec", "class", "confidence", "source")} for a in f_alerts],
        })
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"[eval_audio] {i + 1}/{len(rows)} videos ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- aggregate ----------------
    y = np.concatenate(fl_lab)
    sv, sa = np.concatenate(fl_v), np.concatenate(fl_a)
    frame = {}
    for name, sc in [("video_stage1", sv), ("audio_stage1", sa), ("fused_mean", 0.5 * sv + 0.5 * sa), ("fused_max", np.maximum(sv, sa))]:
        frame[name] = {"roc_auc": round(roc_auc_score(y, sc), 4), "ap": round(average_precision_score(y, sc), 4)}

    sil = {c: {**v, "silent_window_frac": _rate(v["silent_windows"], v["windows"])} for c, v in sorted(silence.items())}

    gt_scored = [g for g in gt_recs if not g["silent"]]
    by_cls = defaultdict(lambda: Counter())
    conf = defaultdict(Counter)
    for g in gt_recs:
        c = by_cls[g["gt_class"]]
        c["windows"] += 1
        c["silent_or_no_audio"] += int(g["silent"])
        if not g["silent"]:
            c["scored"] += 1
            c["top1_correct"] += int(g["pred_video"] == g["gt_class"])
            c["anom_top1_correct"] += int(AUDIO_TO_VIDEO_CLASS.get(g["pred_anom"]) == g["gt_class"])
            c["suppressed"] += int(g["suppressed"])
            conf[g["gt_class"]][g["pred"]] += 1
    gt_summary = {
        "n_windows": len(gt_recs),
        "n_silent_or_no_audio": len(gt_recs) - len(gt_scored),
        "n_scored": len(gt_scored),
        "top1_acc_scored": _rate(sum(g["pred_video"] == g["gt_class"] for g in gt_scored), len(gt_scored)),
        "anom5_top1_acc_scored": _rate(sum(AUDIO_TO_VIDEO_CLASS.get(g["pred_anom"]) == g["gt_class"] for g in gt_scored), len(gt_scored)),
        "suppressed_rate_scored": _rate(sum(g["suppressed"] for g in gt_scored), len(gt_scored)),
        "per_class": {k: dict(v) for k, v in sorted(by_cls.items())},
        "confusion_gt_x_clap": {k: {c: v[c] for c in AUDIO_CLASSES} for k, v in sorted(conf.items())},
    }

    sweep_out = [{"tau_a": t, "trigger_window_rate": _rate(v["flag_windows"], v["windows"]),
                  "anom_triggered": _rate(v["anom_trig"], n_anom), "norm_triggered": _rate(v["norm_trig"], n_norm),
                  "anom_alert_recall": _rate(v["anom_alert"], n_anom), "norm_alert_far": _rate(v["norm_alert"], n_norm)}
                 for t, v in sweep_counts.items()]

    def pv_acc(pairs: list[tuple[str, str]]) -> dict:
        return {"n": len(pairs), "top1": _rate(sum(a == b for a, b in pairs), len(pairs))}

    e2e_out = {m: {"anom_recall": _rate(e2e[m]["anom"], n_anom), "norm_far": _rate(e2e[m]["norm"], n_norm),
                   "anom_alerted": e2e[m]["anom"], "norm_alerted": e2e[m]["norm"]} for m in e2e}
    e2e_out["fused"]["alerts_by_source"] = {k: v for k, v in e2e["fused"].items() if "_" in k}
    e2e_out["per_video_top1"] = {m: pv_acc(p) for m, p in pervid.items()}

    a_sec = max(1e-6, timing["audio_sec"])
    summary = {
        "checkpoint": args.checkpoint, "tau_v": tau_v, "tau_a": args.tau_a, "device": dev,
        "n_videos": len(rows), "n_anomaly": n_anom, "n_normal": n_norm,
        "elapsed_sec": round(time.time() - t_start, 1),
        "timing": {"trigger_ms_per_audio_sec": round(1000 * timing["trigger_sec"] / a_sec, 2),
                   "clap_ms_per_window": round(1000 * timing["clap_sec"] / max(1, timing["clap_windows"]), 1),
                   "audio_hours": round(a_sec / 3600, 2), "decode_sec": round(timing["decode_sec"], 1)},
        "silence_audit": sil,
        "frame_level": frame,
        "audio_stage2_gt_windows": gt_summary,
        "tau_a_sweep": sweep_out,
        "end_to_end": e2e_out,
        "gt_window_records": gt_recs,
        "per_video": per_video_out,
    }
    (out_dir / "audio_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== Silence audit (1 s windows below %.0f dBFS) =====" % SILENCE_DBFS)
    for c, v in sil.items():
        print(f"  {c:10s} videos={v['videos']:3d} no_audio={v['no_audio']:3d} all_silent={v['all_silent_videos']:3d} "
              f"silent_windows={v['silent_window_frac']:.1%}")
    print("\n===== Frame-level over %d test videos =====" % len(rows))
    for k, v in frame.items():
        print(f"  {k:13s} ROC-AUC={v['roc_auc']:.4f}  AP={v['ap']:.4f}")
    print("\n===== Audio Stage 2 (CLAP) on GT anomaly windows =====")
    print(f"  windows={gt_summary['n_windows']} silent/no-audio={gt_summary['n_silent_or_no_audio']} scored={gt_summary['n_scored']} | "
          f"top-1={gt_summary['top1_acc_scored']:.3f} anomaly-5 top-1={gt_summary['anom5_top1_acc_scored']:.3f} "
          f"suppressed={gt_summary['suppressed_rate_scored']:.1%}")
    for c, v in gt_summary["per_class"].items():
        print(f"  {c:10s} {dict(v)}")
    print("\n===== τ_a sweep (audio-only, video level) =====")
    for s_ in sweep_out:
        print(f"  τ_a={s_['tau_a']:.2f} windows>τ={s_['trigger_window_rate']:.1%} recall={s_['anom_alert_recall']:.3f} "
              f"FAR={s_['norm_alert_far']:.3f} (triggered A={s_['anom_triggered']:.3f} N={s_['norm_triggered']:.3f})")
    print("\n===== End to end (τ_v=%.4f, τ_a=%.2f) =====" % (tau_v, args.tau_a))
    for m in ("video", "audio", "fused"):
        v = e2e_out[m]
        print(f"  {m:6s} recall {v['anom_alerted']}/{n_anom} ({v['anom_recall']:.1%})  normal FAR {v['norm_alerted']}/{n_norm} ({v['norm_far']:.1%})")
    print(f"  per-video top-1: video {e2e_out['per_video_top1']['video']}  fused {e2e_out['per_video_top1']['fused']}")
    print(f"  fused alerts by source: {e2e_out['fused']['alerts_by_source']}")
    print(f"\n  timing: {summary['timing']}  elapsed {summary['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
