"""Stage 2 evaluation: top-1/top-3 accuracy and confusion matrix on GT & Stage 1 windows.

Implements Execution plan §3.4:
  - Maps UCF-Crime labels to Stage 2 classes (Arson -> fire, Burglary -> burglary,
    Explosion -> explosion, Fighting -> fighting, Robbery -> robbery, Shooting -> shooting).
  - Evaluates CLIP ViT-B/16 on:
      1. Ground-truth temporal anomaly windows (from Temporal_Anomaly_Annotation_for_Testing_Videos.txt)
      2. Stage 1 flagged windows (above hold-out τ, merged < 2 clips, padded ±1 clip)
      3. Optional false-alarm veto check on normal test videos (before vs after CLIP suppression)
  - Reports top-1 accuracy, top-3 accuracy, and confusion matrix (both all-9 and anomaly-only).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vad.data.ucf_dataset import (
    DEFAULT_FRAMES_PER_CLIP,
    l2_normalise,
    load_manifest,
    load_temporal_annotations,
)
from vad.eval.metrics import classification_metrics, score_clips
from vad.pipeline.run_video import (
    WindowScorer,
    extract_flagged_windows,
    load_stage1_checkpoint,
    sample_window_frames,
)
from vad.stage1.model import MILHead
from vad.stage2.clip_scorer import (
    ANOMALY_CLASSES,
    DEFAULT_CLIP_MODEL,
    DEFAULT_SUPPRESS_THRESHOLD,
    STAGE2_CLASSES,
    UCF_TO_STAGE2_CLASS,
    ClipScorer,
)

UCF_TEST_STAGE2_CLASSES: tuple[str, ...] = (
    "burglary",
    "fire",
    "shooting",
    "fighting",
    "robbery",
    "explosion",
)


def _format_cm(cm: list[list[int]], classes: Sequence[str]) -> str:
    header = "gt \\ pred".ljust(15) + " ".join(f"{c[:8]:>8s}" for c in classes)
    lines = [header, "-" * len(header)]
    for cls_name, row in zip(classes, cm):
        lines.append(cls_name.ljust(15) + " ".join(f"{v:8d}" for v in row))
    return "\n".join(lines)


def evaluate_gt_windows(
    test_anom_rows: Sequence[dict[str, str]],
    annotations: dict[str, tuple[str, tuple[tuple[int, int], ...]]],
    videos_dir: Path,
    scorer: WindowScorer,
    *,
    num_frames: int = 8,
    suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate Stage 2 on ground-truth anomaly windows of UCF test anomaly videos."""
    y_true: list[str] = []
    y_top3_all: list[list[str]] = []
    y_top3_anom: list[list[str]] = []
    suppressed = 0
    records: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    for row in test_anom_rows:
        vid = row["video_id"]
        gt_cls = UCF_TO_STAGE2_CLASS[row["cls"]]
        vid_path = videos_dir / row["rel_path"]
        ivs = annotations.get(vid, ("", ()))[1]
        if not ivs or not vid_path.exists():
            errors.append((vid, "missing video or annotation"))
            continue

        for iv_idx, (s, e) in enumerate(ivs):
            try:
                frames, _, timestamps = sample_window_frames(
                    vid_path, start_frame=max(0, s - 1), end_frame=e, num_frames=num_frames
                )
                res = scorer.score_window(
                    frames,
                    frame_timestamps=timestamps,
                    suppress_threshold=suppress_threshold,
                    top_k=3,
                )
            except Exception as exc:
                errors.append((f"{vid}#iv{iv_idx}", f"{type(exc).__name__}: {exc}"))
                continue

            top3_all = [t["class"] for t in res["top3"]]
            top3_anom = [t["class"] for t in res["top3_anomaly"]]
            y_true.append(gt_cls)
            y_top3_all.append(top3_all)
            y_top3_anom.append(top3_anom)
            suppressed += int(res["suppressed"])
            records.append({
                "video_id": vid,
                "interval": [s, e],
                "gt_class": gt_cls,
                "pred_all9": top3_all[0],
                "top3_all9": res["top3"],
                "pred_anom7": top3_anom[0],
                "top3_anom7": res["top3_anomaly"],
                "suppressed": res["suppressed"],
                "non_anomaly_prob": res["non_anomaly_prob"],
            })

    m_all9 = classification_metrics(y_true, y_top3_all, STAGE2_CLASSES)
    m_anom7 = classification_metrics(y_true, y_top3_anom, ANOMALY_CLASSES)
    m_ucf6 = classification_metrics(y_true, y_top3_anom, UCF_TEST_STAGE2_CLASSES)
    return {
        "n_windows": len(y_true),
        "n_suppressed": suppressed,
        "suppressed_rate": round(suppressed / max(1, len(y_true)), 4),
        "all_9_classes": m_all9,
        "anomaly_7_classes": m_anom7,
        "ucf_6_classes": m_ucf6,
        "errors": errors,
        "records": records,
    }


def evaluate_stage1_windows(
    test_rows: Sequence[dict[str, str]],
    videos_dir: Path,
    features_dir: Path,
    stage1_model: MILHead,
    scorer: WindowScorer,
    tau: float,
    device: torch.device | str,
    *,
    smooth_window: int = 3,
    num_frames: int = 8,
    suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
    eval_normals: bool = True,
) -> dict[str, Any]:
    """Evaluate Stage 2 on Stage 1 flagged windows (anomalies + optional normal veto rate)."""
    y_true_win: list[str] = []
    y_top3_all_win: list[list[str]] = []
    y_top3_anom_win: list[list[str]] = []

    y_true_vid: list[str] = []
    y_top3_all_vid: list[list[str]] = []
    y_top3_anom_vid: list[list[str]] = []

    anom_total = anom_flagged_s1 = anom_alerted_after_clip = 0
    norm_total = norm_flagged_s1 = norm_alerted_after_clip = 0
    anom_windows_suppressed = 0
    norm_windows_total = norm_windows_suppressed = 0
    records: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    for row in test_rows:
        vid = row["video_id"]
        label = int(row["label"])
        if label == 0 and not eval_normals:
            continue
        vid_path = videos_dir / row["rel_path"]
        npy_path = features_dir / f"{vid}.npy"
        json_path = features_dir / f"{vid}.json"
        if not vid_path.exists() or not npy_path.exists():
            errors.append((vid, "missing video or features"))
            continue

        raw_feats = np.load(npy_path)
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        fps = float(meta.get("fps") or 30.0)
        fpc = int(meta.get("stride") or 6) * int(meta.get("clip_len") or 13)
        n_frames = int(meta.get("n_frames") or (raw_feats.shape[0] * fpc))

        _, sm = score_clips(stage1_model, l2_normalise(raw_feats), device=device, smooth_window=smooth_window)
        wins = extract_flagged_windows(sm, tau=tau, n_frames=n_frames, fps=fps, frames_per_clip=fpc)

        if label == 1:
            anom_total += 1
            if wins:
                anom_flagged_s1 += 1
        else:
            norm_total += 1
            if wins:
                norm_flagged_s1 += 1

        if not wins:
            continue

        gt_cls = UCF_TO_STAGE2_CLASS[row["cls"]]
        vid_win_results: list[tuple[float, dict[str, Any]]] = []

        for w in wins:
            try:
                frames, _, timestamps = sample_window_frames(
                    vid_path, start_frame=w.start_frame, end_frame=w.end_frame, num_frames=num_frames, fps=fps
                )
                res = scorer.score_window(
                    frames,
                    frame_timestamps=timestamps,
                    suppress_threshold=suppress_threshold,
                    top_k=3,
                )
            except Exception as exc:
                errors.append((vid, f"{type(exc).__name__}: {exc}"))
                continue

            vid_win_results.append((w.peak_score, res))
            if label == 1:
                y_true_win.append(gt_cls)
                y_top3_all_win.append([t["class"] for t in res["top3"]])
                y_top3_anom_win.append([t["class"] for t in res["top3_anomaly"]])
                anom_windows_suppressed += int(res["suppressed"])
            else:
                norm_windows_total += 1
                norm_windows_suppressed += int(res["suppressed"])

        kept = [(pk, r) for pk, r in vid_win_results if not r["suppressed"]]
        if label == 1:
            if kept:
                anom_alerted_after_clip += 1
            if vid_win_results:
                # Per-video prediction: highest Stage 1 peak among kept windows (else highest overall)
                best_res = max(kept or vid_win_results, key=lambda x: x[0])[1]
                y_true_vid.append(gt_cls)
                y_top3_all_vid.append([t["class"] for t in best_res["top3"]])
                y_top3_anom_vid.append([t["class"] for t in best_res["top3_anomaly"]])
                records.append({
                    "video_id": vid,
                    "gt_class": gt_cls,
                    "n_s1_windows": len(wins),
                    "n_kept_after_clip": len(kept),
                    "pred_all9": best_res["top3"][0]["class"],
                    "pred_anom7": best_res["top3_anomaly"][0]["class"],
                    "top3_all9": best_res["top3"],
                })
        else:
            if kept:
                norm_alerted_after_clip += 1

    return {
        "tau": float(tau),
        "anomaly_videos": {
            "total": anom_total,
            "flagged_by_stage1": anom_flagged_s1,
            "stage1_recall": round(anom_flagged_s1 / max(1, anom_total), 4),
            "alerted_after_clip": anom_alerted_after_clip,
            "end_to_end_recall": round(anom_alerted_after_clip / max(1, anom_total), 4),
            "n_flagged_windows": len(y_true_win),
            "n_windows_suppressed": anom_windows_suppressed,
        },
        "normal_videos": {
            "total": norm_total,
            "flagged_by_stage1": norm_flagged_s1,
            "stage1_video_far": round(norm_flagged_s1 / max(1, norm_total), 4),
            "alerted_after_clip": norm_alerted_after_clip,
            "after_clip_video_far": round(norm_alerted_after_clip / max(1, norm_total), 4),
            "n_flagged_windows": norm_windows_total,
            "n_windows_suppressed": norm_windows_suppressed,
            "window_veto_rate": round(norm_windows_suppressed / max(1, norm_windows_total), 4),
        },
        "per_window_all9": classification_metrics(y_true_win, y_top3_all_win, STAGE2_CLASSES),
        "per_window_anom7": classification_metrics(y_true_win, y_top3_anom_win, ANOMALY_CLASSES),
        "per_video_all9": classification_metrics(y_true_vid, y_top3_all_vid, STAGE2_CLASSES),
        "per_video_anom7": classification_metrics(y_true_vid, y_top3_anom_vid, ANOMALY_CLASSES),
        "per_video_ucf6": classification_metrics(y_true_vid, y_top3_anom_vid, UCF_TEST_STAGE2_CLASSES),
        "errors": errors,
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage 2 CLIP on UCF-Crime GT & Stage 1 windows.")
    ap.add_argument("--videos", required=True, help="UCF videos root (MyDrive/vad/ucf_videos)")
    ap.add_argument("--features", required=True, help="X3D-S features dir (MyDrive/vad/features/x3d_s)")
    ap.add_argument("--checkpoint", required=True, help="Trained Stage 1 MILHead checkpoint (.pt)")
    ap.add_argument("--manifest", default="data/manifests/ucf_subset.csv")
    ap.add_argument("--annotations", default="data/splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--out", required=True, help="Output directory for stage2_eval.json")
    ap.add_argument("--tau", type=float, default=None, help="Override τ (default: checkpoint hold-out τ)")
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--suppress-threshold", type=float, default=DEFAULT_SUPPRESS_THRESHOLD)
    ap.add_argument("--skip-normals", action="store_true", help="Skip normal test video veto evaluation")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on test anomaly videos (quick test)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    t0 = time.time()
    dev = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_manifest(args.manifest, lanes=("L1", "L2")) if r["split"] == "test"]
    anom_rows = [r for r in rows if int(r["label"]) == 1]
    norm_rows = [r for r in rows if int(r["label"]) == 0]
    if args.limit > 0:
        anom_rows = anom_rows[: args.limit]
        norm_rows = norm_rows[: args.limit]

    annotations = load_temporal_annotations(args.annotations)
    stage1_model, ckpt = load_stage1_checkpoint(args.checkpoint, device=dev)
    tau = args.tau if args.tau is not None else float(ckpt["tau_info"]["tau"])
    smooth_window = int(ckpt.get("smooth_window", 3))

    scorer = ClipScorer(model_name=args.clip_model, device=dev, use_realness=False)

    print(f"[stage2_eval] evaluating {len(anom_rows)} anomaly videos on GT windows...", flush=True)
    gt_res = evaluate_gt_windows(
        anom_rows, annotations, Path(args.videos), scorer,
        num_frames=args.num_frames, suppress_threshold=args.suppress_threshold,
    )

    print(f"[stage2_eval] evaluating Stage 1 flagged windows (tau={tau:.4f})...", flush=True)
    s1_res = evaluate_stage1_windows(
        anom_rows + ([] if args.skip_normals else norm_rows),
        Path(args.videos), Path(args.features), stage1_model, scorer, tau=tau, device=dev,
        smooth_window=smooth_window, num_frames=args.num_frames,
        suppress_threshold=args.suppress_threshold, eval_normals=not args.skip_normals,
    )

    summary = {
        "checkpoint": args.checkpoint,
        "clip_model": args.clip_model,
        "tau": tau,
        "elapsed_sec": round(time.time() - t0, 2),
        "gt_windows": gt_res,
        "stage1_windows": s1_res,
    }
    (out_dir / "stage2_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== Stage 2 Evaluation: Ground-Truth Anomaly Windows =====", flush=True)
    print(
        f"  Windows={gt_res['n_windows']} (suppressed={gt_res['n_suppressed']}, {gt_res['suppressed_rate']:.1%}) | "
        f"All-9 Top-1={gt_res['all_9_classes']['top1_accuracy']:.4f}, Top-3={gt_res['all_9_classes']['top3_accuracy']:.4f} | "
        f"Anom-7 Top-1={gt_res['anomaly_7_classes']['top1_accuracy']:.4f}, Top-3={gt_res['anomaly_7_classes']['top3_accuracy']:.4f}",
        flush=True,
    )
    print(_format_cm(gt_res["ucf_6_classes"]["confusion_matrix"], gt_res["ucf_6_classes"]["classes"]), flush=True)

    print("\n===== Stage 2 Evaluation: Stage 1 Flagged Windows =====", flush=True)
    av = s1_res["anomaly_videos"]
    nv = s1_res["normal_videos"]
    print(
        f"  Anom videos: S1 flagged {av['flagged_by_stage1']}/{av['total']} ({av['stage1_recall']:.1%}), "
        f"after CLIP {av['alerted_after_clip']}/{av['total']} ({av['end_to_end_recall']:.1%})",
        flush=True,
    )
    if nv["total"]:
        print(
            f"  Normal videos: S1 flagged {nv['flagged_by_stage1']}/{nv['total']} ({nv['stage1_video_far']:.1%}), "
            f"after CLIP {nv['alerted_after_clip']}/{nv['total']} ({nv['after_clip_video_far']:.1%}; "
            f"window veto={nv['window_veto_rate']:.1%})",
            flush=True,
        )
    print(
        f"  Per-video All-9 Top-1={s1_res['per_video_all9']['top1_accuracy']:.4f}, "
        f"Top-3={s1_res['per_video_all9']['top3_accuracy']:.4f} | "
        f"Anom-7 Top-1={s1_res['per_video_anom7']['top1_accuracy']:.4f}, "
        f"Top-3={s1_res['per_video_anom7']['top3_accuracy']:.4f}",
        flush=True,
    )
    print(_format_cm(s1_res["per_video_ucf6"]["confusion_matrix"], s1_res["per_video_ucf6"]["classes"]), flush=True)


if __name__ == "__main__":
    main()
