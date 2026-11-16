"""Train video-only (I3D), audio-only (VGGish), and late-fusion MIL heads on XD-Violence.

Reuses the exact Stage 1 MILHead, SultaniMILLoss, 32-segment pooling, Adagrad optimiser,
and frame-level AP/ROC-AUC metrics (Execution plan §3.3 & §3.4).

Usage:
  python -m vad.stage1.train_xdv \
      --xdv-root /content/drive/MyDrive/vad/xdv \
      --out /content/drive/MyDrive/vad/runs/xdv_audio_visual
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from vad.data.ucf_dataset import DEFAULT_NUM_SEGMENTS, pool_to_segments
from vad.data.xdv_dataset import XDV_FRAMES_PER_SNIPPET, XDVVideoRecord, load_xdv_splits
from vad.eval.metrics import (
    average_precision_score,
    clips_to_frames,
    roc_auc_score,
    score_clips,
    smooth_scores,
)
from vad.stage1.model import MILHead, SultaniMILLoss
from vad.stage1.train import _agg, _sample_indices, pick_device, set_seed


def _get_feat(rec: XDVVideoRecord, modality: str) -> np.ndarray:
    arr = rec.video_features if modality == "video" else rec.audio_features
    if arr is None:
        raise ValueError(f"Record {rec.video_id} missing {modality} features")
    return arr


@torch.inference_mode()
def _eval_holdout_video_auc(
    model: MILHead,
    records: Sequence[XDVVideoRecord],
    modality: str,
    device: torch.device,
    smooth_window: int = 3,
) -> tuple[float, float]:
    model.eval()
    y_true = np.array([r.label for r in records], dtype=np.int32)
    peaks = np.empty(len(records), dtype=np.float64)
    for i, rec in enumerate(records):
        _, sm = score_clips(model, _get_feat(rec, modality), device=device, smooth_window=smooth_window)
        peaks[i] = float(sm.max())
    return roc_auc_score(y_true, peaks), average_precision_score(y_true, peaks)


def _train_modality_head(
    modality: str,
    seed: int,
    train_fit: Sequence[XDVVideoRecord],
    holdout: Sequence[XDVVideoRecord],
    device: torch.device,
    *,
    steps: int = 3000,
    batch_size: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    lambda_smooth: float = 8e-5,
    lambda_sparse: float = 8e-5,
    dropout: float = 0.6,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
    smooth_window: int = 3,
    eval_every: int = 100,
    log_every: int = 500,
) -> tuple[MILHead, dict[str, object]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    anom_recs = [r for r in train_fit if r.label == 1]
    norm_recs = [r for r in train_fit if r.label == 0]
    in_dim = _get_feat(anom_recs[0], modality).shape[1]

    anom_bags = torch.from_numpy(
        np.stack([pool_to_segments(_get_feat(r, modality), num_segments) for r in anom_recs], axis=0)
    ).to(device)
    norm_bags = torch.from_numpy(
        np.stack([pool_to_segments(_get_feat(r, modality), num_segments) for r in norm_recs], axis=0)
    ).to(device)

    model = MILHead(in_dim=in_dim, dropout=dropout).to(device)
    criterion = SultaniMILLoss(lambda_smooth=lambda_smooth, lambda_sparse=lambda_sparse)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_key = (-1.0, -1.0)
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        a_idx = _sample_indices(len(anom_recs), batch_size, rng)
        n_idx = _sample_indices(len(norm_recs), batch_size, rng)

        optimizer.zero_grad(set_to_none=True)
        s_a = model(anom_bags[a_idx])
        s_n = model(norm_bags[n_idx])
        loss, parts = criterion(s_a, s_n)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            ho_auc, ho_ap = _eval_holdout_video_auc(model, holdout, modality, device, smooth_window)
            if (ho_auc, ho_ap) > best_key:
                best_key = (ho_auc, ho_ap)
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step == 1 or step % log_every == 0 or step == steps:
                print(
                    f"[xdv:{modality} seed={seed}] step {step:4d}/{steps}  "
                    f"loss={parts['total']:.4f}  holdout_auc={ho_auc:.4f} (best={best_key[0]:.4f}@{best_step})",
                    flush=True,
                )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "modality": modality,
        "in_dim": in_dim,
        "best_step": best_step,
        "holdout_video_auc": best_key[0],
        "holdout_video_ap": best_key[1],
        "elapsed_sec": round(time.time() - t0, 2),
        "state_dict": best_state,
    }


@torch.inference_mode()
def evaluate_xdv_test(
    video_model: MILHead | None,
    audio_model: MILHead | None,
    test_records: Sequence[XDVVideoRecord],
    device: torch.device,
    *,
    smooth_window: int = 3,
    frames_per_snippet: int = XDV_FRAMES_PER_SNIPPET,
) -> dict[str, dict[str, float]]:
    """Compute frame-level AP (primary XD-Violence metric) and ROC-AUC for video, audio, and late fusion."""
    all_gt: list[np.ndarray] = []
    v_frames: list[np.ndarray] = []
    a_frames: list[np.ndarray] = []
    f_frames: list[np.ndarray] = []

    for rec in test_records:
        assert rec.frame_labels is not None
        all_gt.append(rec.frame_labels)

        sv = sa = None
        if video_model is not None and rec.video_features is not None:
            _, sv = score_clips(video_model, rec.video_features, device=device, smooth_window=smooth_window)
            v_frames.append(clips_to_frames(sv, rec.n_frames, frames_per_snippet))
        if audio_model is not None and rec.audio_features is not None:
            _, sa = score_clips(audio_model, rec.audio_features, device=device, smooth_window=smooth_window)
            a_frames.append(clips_to_frames(sa, rec.n_frames, frames_per_snippet))
        if sv is not None and sa is not None:
            # Late fusion: score average per snippet, then mapped to frames
            sf = smooth_scores(0.5 * (sv + sa), window=1)
            f_frames.append(clips_to_frames(sf, rec.n_frames, frames_per_snippet))

    y_true = np.concatenate(all_gt)
    out: dict[str, dict[str, float]] = {}
    if v_frames:
        yv = np.concatenate(v_frames)
        out["video_only"] = {
            "frame_ap": average_precision_score(y_true, yv),
            "frame_roc_auc": roc_auc_score(y_true, yv),
        }
    if a_frames:
        ya = np.concatenate(a_frames)
        out["audio_only"] = {
            "frame_ap": average_precision_score(y_true, ya),
            "frame_roc_auc": roc_auc_score(y_true, ya),
        }
    if f_frames:
        yf = np.concatenate(f_frames)
        out["late_fusion"] = {
            "frame_ap": average_precision_score(y_true, yf),
            "frame_roc_auc": roc_auc_score(y_true, yf),
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train XD-Violence video-only, audio-only, and late-fusion MIL heads.")
    ap.add_argument("--xdv-root", required=True, help="Root folder containing i3d_rgb/, vggish/, annotations.txt")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints and summary.json")
    ap.add_argument("--seeds", default="415,416,417")
    ap.add_argument("--split-seed", type=int, default=415)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lambda-smooth", type=float, default=8e-5)
    ap.add_argument("--lambda-sparse", type=float, default=8e-5)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--smooth-window", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    print(f"[xdv] loading paired I3D RGB + VGGish splits from {args.xdv_root} on {device}...", flush=True)
    train_fit, holdout, test_records = load_xdv_splits(
        args.xdv_root, val_frac=args.val_frac, split_seed=args.split_seed
    )
    print(
        f"[xdv] splits: train_fit={len(train_fit)} ({sum(r.label==1 for r in train_fit)} viol, "
        f"{sum(r.label==0 for r in train_fit)} norm) | holdout={len(holdout)} | test={len(test_records)}",
        flush=True,
    )

    per_seed = []
    for seed in seeds:
        v_model, v_info = _train_modality_head(
            "video", seed, train_fit, holdout, device,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, lambda_smooth=args.lambda_smooth,
            lambda_sparse=args.lambda_sparse, dropout=args.dropout,
            smooth_window=args.smooth_window, eval_every=args.eval_every,
            log_every=args.log_every,
        )
        a_model, a_info = _train_modality_head(
            "audio", seed, train_fit, holdout, device,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, lambda_smooth=args.lambda_smooth,
            lambda_sparse=args.lambda_sparse, dropout=args.dropout,
            smooth_window=args.smooth_window, eval_every=args.eval_every,
            log_every=args.log_every,
        )

        test_res = evaluate_xdv_test(
            v_model, a_model, test_records, device=device, smooth_window=args.smooth_window
        )
        for mod, info in (("video", v_info), ("audio", a_info)):
            torch.save(
                {k: v for k, v in info.items()},
                out_dir / f"xdv_{mod}_seed{seed}.pt",
            )

        rec = {
            "seed": seed,
            "video_best_step": v_info["best_step"],
            "audio_best_step": a_info["best_step"],
            "video_holdout_auc": v_info["holdout_video_auc"],
            "audio_holdout_auc": a_info["holdout_video_auc"],
            "test": test_res,
        }
        per_seed.append(rec)
        print(
            f"[xdv seed={seed} DONE] AP: video={test_res['video_only']['frame_ap']:.4f}  "
            f"audio={test_res['audio_only']['frame_ap']:.4f}  "
            f"late_fusion={test_res['late_fusion']['frame_ap']:.4f} | "
            f"AUC: video={test_res['video_only']['frame_roc_auc']:.4f}  "
            f"audio={test_res['audio_only']['frame_roc_auc']:.4f}  "
            f"late_fusion={test_res['late_fusion']['frame_roc_auc']:.4f}",
            flush=True,
        )

    agg = {
        branch: {
            metric: _agg([r["test"][branch][metric] for r in per_seed])
            for metric in ("frame_ap", "frame_roc_auc")
        }
        for branch in ("video_only", "audio_only", "late_fusion")
    }
    summary = {
        "config": vars(args),
        "split_counts": {"train_fit": len(train_fit), "holdout": len(holdout), "test": len(test_records)},
        "seeds": seeds,
        "summary": agg,
        "per_seed": per_seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== XD-Violence Summary (mean ± std across seeds) =====", flush=True)
    for branch in ("video_only", "audio_only", "late_fusion"):
        ap_m, ap_s = agg[branch]["frame_ap"]["mean"], agg[branch]["frame_ap"]["std"]
        auc_m, auc_s = agg[branch]["frame_roc_auc"]["mean"], agg[branch]["frame_roc_auc"]["std"]
        print(f"  {branch:12s}: AP = {ap_m:.4f} ± {ap_s:.4f}  |  ROC-AUC = {auc_m:.4f} ± {auc_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
