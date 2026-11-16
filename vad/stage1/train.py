"""Train and evaluate the Stage 1 Sultani MIL head on pre-extracted X3D-S features.

Implements Execution plan §3.4 and Stage 1 MIL explained §5–8:
  - Input: L2-normalised clip features pooled to 32 segments during training
  - Model: MILHead (2048 -> 512 ReLU drop 0.6 -> 32 drop 0.6 -> 1 sigmoid)
  - Loss: all-pairs Sultani hinge ranking + 8e-5 smoothness + 8e-5 sparsity
  - Optimiser: Adagrad (lr=1e-3, weight_decay=1e-3)
  - Batch: 30 anomaly + 30 normal videos per step, up to 3,000 steps
  - Hold-out: stratified 10% of train (37 anomaly + 40 normal) for best checkpoint
    (by video-level AUC) and alert threshold τ (>= 90% held-out anomaly recall)
  - Seeds: 3 runs (default 415,416,417), reporting mean ± std on the 226 test videos

Usage:
  python -m vad.stage1.train \
      --features /content/drive/MyDrive/vad/features/x3d_s \
      --manifest data/manifests/ucf_subset.csv \
      --annotations data/splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
      --out /content/drive/MyDrive/vad/runs/stage1_sultani
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from vad.data.ucf_dataset import (
    DEFAULT_FEAT_DIM,
    DEFAULT_NUM_SEGMENTS,
    VideoRecord,
    load_ucf_splits,
    pool_to_segments,
)
from vad.eval.metrics import (
    evaluate_stage1_test,
    evaluate_video_level,
    select_threshold,
)
from vad.stage1.model import MILHead, SultaniMILLoss


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_indices(n_total: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    replace = n_total < batch_size
    return rng.choice(n_total, size=batch_size, replace=replace)


def train_single_seed(
    seed: int,
    train_fit: Sequence[VideoRecord],
    holdout: Sequence[VideoRecord],
    test_records: Sequence[VideoRecord],
    out_dir: Path,
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
    eval_every: int = 50,
    target_recall: float = 0.90,
    log_every: int = 250,
) -> dict[str, object]:
    """Train one seed, checkpoint by hold-out video AUC, pick τ on hold-out, evaluate test."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    anom_recs = [r for r in train_fit if r.label == 1]
    norm_recs = [r for r in train_fit if r.label == 0]
    if not anom_recs or not norm_recs:
        raise ValueError(f"Need both anomaly ({len(anom_recs)}) and normal ({len(norm_recs)}) training videos")

    in_dim = anom_recs[0].features.shape[1]
    anom_bags = torch.from_numpy(
        np.stack([pool_to_segments(r.features, num_segments) for r in anom_recs], axis=0)
    ).to(device)  # [N_a, 32, D]
    norm_bags = torch.from_numpy(
        np.stack([pool_to_segments(r.features, num_segments) for r in norm_recs], axis=0)
    ).to(device)  # [N_n, 32, D]

    model = MILHead(in_dim=in_dim, dropout=dropout).to(device)
    criterion = SultaniMILLoss(lambda_smooth=lambda_smooth, lambda_sparse=lambda_sparse)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_holdout_key = (-1.0, -1.0)
    best_holdout_auc = -1.0
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        a_idx = _sample_indices(len(anom_recs), batch_size, rng)
        n_idx = _sample_indices(len(norm_recs), batch_size, rng)

        x_a = anom_bags[a_idx]  # [B, 32, D]
        x_n = norm_bags[n_idx]  # [B, 32, D]

        optimizer.zero_grad(set_to_none=True)
        s_a = model(x_a)
        s_n = model(x_n)
        loss, parts = criterion(s_a, s_n)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            ho = evaluate_video_level(model, holdout, device=device, smooth_window=smooth_window)
            ho_auc = float(ho["video_auc"])
            ho_ap = float(ho["video_ap"])
            entry = {"step": step, "holdout_video_auc": ho_auc, "holdout_video_ap": ho_ap, **parts}
            history.append(entry)
            if (ho_auc, ho_ap) > best_holdout_key:
                best_holdout_key = (ho_auc, ho_ap)
                best_holdout_auc = ho_auc
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if step == 1 or step % log_every == 0 or step == steps:
                print(
                    f"[seed {seed}] step {step:4d}/{steps}  "
                    f"loss={parts['total']:.4f} (rank={parts['ranking']:.4f} "
                    f"sm={parts['smoothness']:.4f} sp={parts['sparsity']:.4f})  "
                    f"max_a={parts['mean_max_anom']:.3f} max_n={parts['mean_max_norm']:.3f}  "
                    f"holdout_auc={ho_auc:.4f} (best={best_holdout_auc:.4f}@{best_step})",
                    flush=True,
                )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device).eval()

    # Choose τ strictly on the held-out train split
    ho_best = evaluate_video_level(model, holdout, device=device, smooth_window=smooth_window)
    tau_info = select_threshold(
        anom_peaks=ho_best["anom_peaks"],
        norm_peaks=ho_best["norm_peaks"],
        target_recall=target_recall,
        norm_clip_scores=ho_best["norm_clip_scores"],
    )
    tau = float(tau_info["tau"])

    # Final evaluation on the 226 test videos
    test_metrics = evaluate_stage1_test(
        model,
        test_records,
        device=device,
        smooth_window=smooth_window,
        num_segments=num_segments,
        tau=tau,
    )

    elapsed = round(time.time() - t0, 2)
    ckpt_path = out_dir / f"best_seed{seed}.pt"
    torch.save(
        {
            "seed": seed,
            "best_step": best_step,
            "in_dim": in_dim,
            "hidden1": model.hidden1,
            "hidden2": model.hidden2,
            "dropout": dropout,
            "num_segments": num_segments,
            "smooth_window": smooth_window,
            "state_dict": best_state,
            "holdout_video_auc": best_holdout_auc,
            "tau_info": tau_info,
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )

    seed_result = {
        "seed": seed,
        "best_step": best_step,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path),
        "holdout_video_auc": best_holdout_auc,
        "holdout_video_ap": float(ho_best["video_ap"]),
        "tau_info": tau_info,
        "test_metrics": test_metrics,
        "initial_loss": history[0]["total"],
        "final_loss": history[-1]["total"],
        "history": history,
    }
    (out_dir / f"seed{seed}_results.json").write_text(
        json.dumps(seed_result, indent=2), encoding="utf-8"
    )
    print(
        f"[seed {seed} DONE] {elapsed:.1f}s | best_step={best_step} | "
        f"test frame AUC={test_metrics['frame_roc_auc']:.4f}  AP={test_metrics['frame_ap']:.4f} | "
        f"32-seg AUC={test_metrics['seg32_frame_roc_auc']:.4f}  AP={test_metrics['seg32_frame_ap']:.4f} | "
        f"tau={tau:.4f} (holdout rec={tau_info['holdout_anom_recall']:.3f}, "
        f"norm FAR={tau_info['holdout_norm_video_far']:.3f})",
        flush=True,
    )
    return seed_result


def _agg(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_seeds(seed_results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate mean ± std across seeds."""
    metrics_keys = (
        "frame_roc_auc",
        "frame_ap",
        "frame_roc_auc_raw",
        "frame_ap_raw",
        "seg32_frame_roc_auc",
        "seg32_frame_ap",
        "test_video_roc_auc",
        "test_anom_recall_at_tau",
        "test_norm_video_far_at_tau",
        "test_norm_clip_far_at_tau",
    )
    agg_test = {
        k: _agg([float(r["test_metrics"][k]) for r in seed_results]) for k in metrics_keys
    }
    agg_holdout = {
        "holdout_video_auc": _agg([float(r["holdout_video_auc"]) for r in seed_results]),
        "tau": _agg([float(r["tau_info"]["tau"]) for r in seed_results]),
        "holdout_anom_recall": _agg([float(r["tau_info"]["holdout_anom_recall"]) for r in seed_results]),
        "holdout_norm_video_far": _agg([float(r["tau_info"]["holdout_norm_video_far"]) for r in seed_results]),
        "holdout_norm_clip_far": _agg([float(r["tau_info"].get("holdout_norm_clip_far", 0.0)) for r in seed_results]),
    }
    return {
        "n_seeds": len(seed_results),
        "seeds": [int(r["seed"]) for r in seed_results],
        "test": agg_test,
        "holdout": agg_holdout,
        "per_seed": [
            {
                "seed": r["seed"],
                "best_step": r["best_step"],
                "elapsed_sec": r["elapsed_sec"],
                "holdout_video_auc": r["holdout_video_auc"],
                "tau_info": r["tau_info"],
                "test_metrics": r["test_metrics"],
                "checkpoint": r["checkpoint"],
            }
            for r in seed_results
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train Stage 1 Sultani MIL head on UCF-Crime X3D-S features.")
    ap.add_argument("--features", required=True, help="Directory with <video_id>.npy and .json files")
    ap.add_argument("--manifest", default="data/manifests/ucf_subset.csv")
    ap.add_argument("--annotations", default="data/splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--lanes", default="L1,L2", help="Comma-separated lanes (default L1,L2; L2X stays off)")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints and JSON metrics")
    ap.add_argument("--seeds", default="415,416,417", help="Comma-separated training seeds")
    ap.add_argument("--split-seed", type=int, default=415, help="Deterministic seed for stratified hold-out split")
    ap.add_argument("--val-frac", type=float, default=0.10, help="Stratified hold-out fraction of train videos")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=30, help="Anomaly videos + normal videos per step (30+30=60)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lambda-smooth", type=float, default=8e-5)
    ap.add_argument("--lambda-sparse", type=float, default=8e-5)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--num-segments", type=int, default=DEFAULT_NUM_SEGMENTS)
    ap.add_argument("--smooth-window", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--feat-dim", type=int, default=DEFAULT_FEAT_DIM)
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--skip-missing", action="store_true", help="Skip manifest videos missing .npy (smoke tests only)")
    args = ap.parse_args(argv)

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    print(f"[stage1] device={device} lanes={lanes} seeds={seeds} steps={args.steps}", flush=True)
    train_fit, holdout, test_records = load_ucf_splits(
        features_dir=args.features,
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        lanes=lanes,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        expected_dim=args.feat_dim,
        skip_missing=args.skip_missing,
    )

    n_tf_a = sum(r.label == 1 for r in train_fit)
    n_tf_n = sum(r.label == 0 for r in train_fit)
    n_ho_a = sum(r.label == 1 for r in holdout)
    n_ho_n = sum(r.label == 0 for r in holdout)
    n_te_a = sum(r.label == 1 for r in test_records)
    n_te_n = sum(r.label == 0 for r in test_records)
    print(
        f"[stage1] splits: train_fit={len(train_fit)} ({n_tf_a} anom, {n_tf_n} norm) | "
        f"holdout={len(holdout)} ({n_ho_a} anom, {n_ho_n} norm) | "
        f"test={len(test_records)} ({n_te_a} anom, {n_te_n} norm)",
        flush=True,
    )
    sample_head = MILHead(in_dim=args.feat_dim, dropout=args.dropout)
    print(f"[stage1] MILHead parameters: {sample_head.num_parameters():,}", flush=True)

    # Save exact hold-out video IDs for full reproducibility
    (out_dir / "split_ids.json").write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "val_frac": args.val_frac,
                "lanes": lanes,
                "train_fit": [r.video_id for r in train_fit],
                "holdout": [r.video_id for r in holdout],
                "test": [r.video_id for r in test_records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seed_results = []
    for seed in seeds:
        res = train_single_seed(
            seed=seed,
            train_fit=train_fit,
            holdout=holdout,
            test_records=test_records,
            out_dir=out_dir,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_smooth=args.lambda_smooth,
            lambda_sparse=args.lambda_sparse,
            dropout=args.dropout,
            num_segments=args.num_segments,
            smooth_window=args.smooth_window,
            eval_every=args.eval_every,
            target_recall=args.target_recall,
            log_every=args.log_every,
        )
        seed_results.append(res)

    summary = {
        "config": vars(args),
        "split_counts": {
            "train_fit": {"total": len(train_fit), "anom": n_tf_a, "norm": n_tf_n},
            "holdout": {"total": len(holdout), "anom": n_ho_a, "norm": n_ho_n},
            "test": {"total": len(test_records), "anom": n_te_a, "norm": n_te_n},
        },
        "model_parameters": sample_head.num_parameters(),
        **summarize_seeds(seed_results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t_agg = summary["test"]
    h_agg = summary["holdout"]
    print("\n===== Stage 1 Summary (mean ± std across seeds) =====", flush=True)
    print(
        f"  Frame ROC-AUC (per-clip, 3-clip MA): "
        f"{t_agg['frame_roc_auc']['mean']:.4f} ± {t_agg['frame_roc_auc']['std']:.4f}",
        flush=True,
    )
    print(
        f"  Frame AP      (per-clip, 3-clip MA): "
        f"{t_agg['frame_ap']['mean']:.4f} ± {t_agg['frame_ap']['std']:.4f}",
        flush=True,
    )
    print(
        f"  32-segment check ROC-AUC / AP:       "
        f"{t_agg['seg32_frame_roc_auc']['mean']:.4f} ± {t_agg['seg32_frame_roc_auc']['std']:.4f}  /  "
        f"{t_agg['seg32_frame_ap']['mean']:.4f} ± {t_agg['seg32_frame_ap']['std']:.4f}",
        flush=True,
    )
    print(
        f"  Hold-out video AUC:                  "
        f"{h_agg['holdout_video_auc']['mean']:.4f} ± {h_agg['holdout_video_auc']['std']:.4f}",
        flush=True,
    )
    print(
        f"  Hold-out τ / normal video FAR:       "
        f"{h_agg['tau']['mean']:.4f} ± {h_agg['tau']['std']:.4f}  /  "
        f"{h_agg['holdout_norm_video_far']['mean']:.4f} ± {h_agg['holdout_norm_video_far']['std']:.4f}",
        flush=True,
    )
    print(f"Saved summary to {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
