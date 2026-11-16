"""Stage 1 & Stage 2 evaluation metrics, score smoothing, frame mapping, and τ selection.

Implements Execution plan §3.4 and Stage 1 MIL explained §8:
  - Per-clip scoring with 3-clip moving average smoothing
  - Frame mapping: frame f -> clip floor(f / 78)
  - 32-segment pooling check on test videos
  - Exact tie-aware frame-level ROC-AUC and Average Precision (pure NumPy, matches sklearn)
  - Hold-out threshold τ selection (>= 90% held-out anomaly recall) + normal FAR
  - Multi-class top-1 / top-3 accuracy and confusion matrix (for Stage 2)
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from vad.data.ucf_dataset import (
    DEFAULT_FRAMES_PER_CLIP,
    DEFAULT_NUM_SEGMENTS,
    VideoRecord,
    pool_to_segments,
)


def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative (fps, tps) at each distinct descending score threshold."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("Empty labels/scores")

    y_bin = (y_true > 0).astype(np.int64)
    n_pos = int(y_bin.sum())
    if n_pos == 0 or n_pos == y_bin.size:
        raise ValueError(f"ROC-AUC/AP require both positive and negative labels (pos={n_pos}/{y_bin.size})")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_bin[order]
    s_sorted = y_score[order]

    distinct = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.concatenate((distinct, [y_sorted.size - 1]))

    tps = np.cumsum(y_sorted, dtype=np.float64)[threshold_idxs]
    fps = (1.0 + threshold_idxs.astype(np.float64)) - tps
    return fps, tps


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware binary ROC-AUC (matches sklearn.metrics.roc_auc_score)."""
    fps, tps = _binary_clf_curve(y_true, y_score)
    tpr = np.concatenate(([0.0], tps / tps[-1]))
    fpr = np.concatenate(([0.0], fps / fps[-1]))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware Average Precision (matches sklearn.metrics.average_precision_score)."""
    fps, tps = _binary_clf_curve(y_true, y_score)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def smooth_scores(scores: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving average over `window` clips without zero-padding boundary drop."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    n = arr.size
    if window <= 1 or n <= 1:
        return arr.astype(np.float32)
    half_left = (window - 1) // 2
    half_right = window - 1 - half_left
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    starts = np.maximum(0, np.arange(n) - half_left)
    ends = np.minimum(n, np.arange(n) + half_right + 1)
    out = (csum[ends] - csum[starts]) / (ends - starts)
    return out.astype(np.float32)


def clips_to_frames(
    clip_scores: np.ndarray,
    n_frames: int,
    frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP,
) -> np.ndarray:
    """Map per-clip scores to frame-level scores: frame f -> clip floor(f / 78)."""
    scores = np.asarray(clip_scores, dtype=np.float32).ravel()
    if scores.size == 0 or n_frames <= 0:
        raise ValueError(f"Invalid clip_scores ({scores.size}) or n_frames ({n_frames})")
    idx = np.minimum(np.arange(n_frames, dtype=np.int64) // frames_per_clip, scores.size - 1)
    return scores[idx]


def segments_to_frames(seg_scores: np.ndarray, n_frames: int) -> np.ndarray:
    """Map fixed-segment (e.g. 32-segment) scores evenly across `n_frames`."""
    scores = np.asarray(seg_scores, dtype=np.float32).ravel()
    if scores.size == 0 or n_frames <= 0:
        raise ValueError(f"Invalid seg_scores ({scores.size}) or n_frames ({n_frames})")
    idx = np.minimum((np.arange(n_frames, dtype=np.int64) * scores.size) // n_frames, scores.size - 1)
    return scores[idx]


@torch.inference_mode()
def score_clips(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device | str,
    smooth_window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Score L2-normalised [n_clips, D] features. Returns (raw_scores, smoothed_scores)."""
    model.eval()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    raw = model(x).detach().cpu().numpy().astype(np.float32).ravel()
    smoothed = smooth_scores(raw, window=smooth_window)
    return raw, smoothed


@torch.inference_mode()
def evaluate_video_level(
    model: torch.nn.Module,
    records: Sequence[VideoRecord],
    device: torch.device | str,
    smooth_window: int = 3,
) -> dict[str, object]:
    """Compute video-level ROC-AUC (max smoothed per-clip score vs binary video label)."""
    model.eval()
    labels: list[int] = []
    peaks: list[float] = []
    anom_peaks: list[float] = []
    norm_peaks: list[float] = []
    norm_clips: list[np.ndarray] = []

    for rec in records:
        _, sm = score_clips(model, rec.features, device=device, smooth_window=smooth_window)
        pk = float(sm.max())
        labels.append(rec.label)
        peaks.append(pk)
        if rec.label == 1:
            anom_peaks.append(pk)
        else:
            norm_peaks.append(pk)
            norm_clips.append(sm)

    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(peaks, dtype=np.float64)
    return {
        "video_auc": roc_auc_score(y_true, y_score),
        "video_ap": average_precision_score(y_true, y_score),
        "anom_peaks": np.asarray(anom_peaks, dtype=np.float64),
        "norm_peaks": np.asarray(norm_peaks, dtype=np.float64),
        "norm_clip_scores": np.concatenate(norm_clips) if norm_clips else np.empty(0, dtype=np.float32),
    }


def select_threshold(
    anom_peaks: np.ndarray,
    norm_peaks: np.ndarray,
    target_recall: float = 0.90,
    norm_clip_scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Choose alert threshold τ on held-out train videos (never on test).

    Selects the tightest (highest) threshold τ at which at least `target_recall`
    of held-out anomaly videos peak strictly above τ, and reports the resulting
    false-alarm rate on held-out normal videos (and normal clips if provided).
    """
    anom = np.asarray(anom_peaks, dtype=np.float64).ravel()
    norm = np.asarray(norm_peaks, dtype=np.float64).ravel()
    if anom.size == 0 or norm.size == 0:
        raise ValueError("Both anomaly and normal peaks are required to select τ")

    sorted_anom = np.sort(anom)
    needed = min(anom.size, max(1, math.ceil(target_recall * anom.size)))
    idx = anom.size - needed

    if idx == 0:
        tau = float(max(0.0, sorted_anom[0] - 1e-6))
    elif sorted_anom[idx - 1] < sorted_anom[idx]:
        tau = float(0.5 * (sorted_anom[idx - 1] + sorted_anom[idx]))
    else:
        tau = float(max(0.0, sorted_anom[idx] - 1e-6))

    anom_flagged = int(np.sum(anom > tau))
    norm_flagged = int(np.sum(norm > tau))
    result: dict[str, float | int] = {
        "tau": tau,
        "target_recall": float(target_recall),
        "holdout_anom_recall": float(anom_flagged / anom.size),
        "holdout_anom_flagged": anom_flagged,
        "holdout_anom_total": int(anom.size),
        "holdout_norm_video_far": float(norm_flagged / norm.size),
        "holdout_norm_flagged": norm_flagged,
        "holdout_norm_total": int(norm.size),
    }
    if norm_clip_scores is not None and len(norm_clip_scores) > 0:
        nc = np.asarray(norm_clip_scores, dtype=np.float64).ravel()
        result["holdout_norm_clip_far"] = float(np.mean(nc > tau))
    return result


@torch.inference_mode()
def evaluate_stage1_test(
    model: torch.nn.Module,
    test_records: Sequence[VideoRecord],
    device: torch.device | str,
    smooth_window: int = 3,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
    tau: float | None = None,
) -> dict[str, float | int]:
    """Evaluate Stage 1 head on all test videos concatenated.

    Reports:
      - Primary: per-clip + 3-clip moving average frame-level ROC-AUC and AP
      - Diagnostic: unsmoothed per-clip frame-level ROC-AUC and AP
      - 32-segment check: pooled 32-segment frame-level ROC-AUC and AP
      - Video-level ROC-AUC and (if tau is provided) test-set recall & normal FAR at tau
    """
    model.eval()
    all_labels: list[np.ndarray] = []
    all_clip_smooth: list[np.ndarray] = []
    all_clip_raw: list[np.ndarray] = []
    all_seg32: list[np.ndarray] = []

    vid_labels: list[int] = []
    vid_peaks: list[float] = []
    anom_peaks: list[float] = []
    norm_peaks: list[float] = []
    norm_clip_scores: list[np.ndarray] = []

    for rec in test_records:
        if rec.frame_labels is None:
            raise ValueError(f"Test video {rec.video_id} is missing frame_labels")

        raw, sm = score_clips(model, rec.features, device=device, smooth_window=smooth_window)
        all_labels.append(rec.frame_labels)
        all_clip_raw.append(clips_to_frames(raw, rec.n_frames, rec.frames_per_clip))
        all_clip_smooth.append(clips_to_frames(sm, rec.n_frames, rec.frames_per_clip))

        # 32-segment check
        seg_feats = pool_to_segments(rec.features, num_segments=num_segments)
        seg_raw, _ = score_clips(model, seg_feats, device=device, smooth_window=1)
        all_seg32.append(segments_to_frames(seg_raw, rec.n_frames))

        pk = float(sm.max())
        vid_labels.append(rec.label)
        vid_peaks.append(pk)
        if rec.label == 1:
            anom_peaks.append(pk)
        else:
            norm_peaks.append(pk)
            norm_clip_scores.append(sm)

    y_true = np.concatenate(all_labels)
    y_smooth = np.concatenate(all_clip_smooth)
    y_raw = np.concatenate(all_clip_raw)
    y_seg32 = np.concatenate(all_seg32)

    out: dict[str, float | int] = {
        "n_test_videos": len(test_records),
        "n_test_frames": int(y_true.size),
        "n_anom_frames": int(y_true.sum()),
        "frame_roc_auc": roc_auc_score(y_true, y_smooth),
        "frame_ap": average_precision_score(y_true, y_smooth),
        "frame_roc_auc_raw": roc_auc_score(y_true, y_raw),
        "frame_ap_raw": average_precision_score(y_true, y_raw),
        "seg32_frame_roc_auc": roc_auc_score(y_true, y_seg32),
        "seg32_frame_ap": average_precision_score(y_true, y_seg32),
        "test_video_roc_auc": roc_auc_score(np.asarray(vid_labels), np.asarray(vid_peaks)),
    }
    if tau is not None:
        ap_arr = np.asarray(anom_peaks, dtype=np.float64)
        np_arr = np.asarray(norm_peaks, dtype=np.float64)
        nc_arr = np.concatenate(norm_clip_scores) if norm_clip_scores else np.empty(0)
        out["tau"] = float(tau)
        out["test_anom_recall_at_tau"] = float(np.mean(ap_arr > tau)) if ap_arr.size else 0.0
        out["test_norm_video_far_at_tau"] = float(np.mean(np_arr > tau)) if np_arr.size else 0.0
        out["test_norm_clip_far_at_tau"] = float(np.mean(nc_arr > tau)) if nc_arr.size else 0.0
    return out


def classification_metrics(
    y_true: Sequence[str],
    y_top3: Sequence[Sequence[str]],
    classes: Sequence[str],
) -> dict[str, object]:
    """Compute top-1 accuracy, top-3 accuracy, and confusion matrix for Stage 2."""
    if len(y_true) != len(y_top3):
        raise ValueError("y_true and y_top3 must have the same length")
    n = len(y_true)
    cls_list = list(classes)
    cls_to_idx = {c: i for i, c in enumerate(cls_list)}
    cm = np.zeros((len(cls_list), len(cls_list)), dtype=np.int64)

    top1_hits = 0
    top3_hits = 0
    for gt, preds in zip(y_true, y_top3):
        p_list = list(preds)
        top1 = p_list[0] if p_list else None
        if top1 == gt:
            top1_hits += 1
        if gt in p_list[:3]:
            top3_hits += 1
        if gt in cls_to_idx and top1 in cls_to_idx:
            cm[cls_to_idx[gt], cls_to_idx[top1]] += 1

    return {
        "n_samples": n,
        "top1_accuracy": float(top1_hits / n) if n else 0.0,
        "top3_accuracy": float(top3_hits / n) if n else 0.0,
        "classes": cls_list,
        "confusion_matrix": cm.tolist(),
    }
