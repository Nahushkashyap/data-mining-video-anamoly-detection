"""UCF-Crime X3D-S feature loader, 32-segment pooling, frame labels, and hold-out split.

Implements the approved Stage 1 data pipeline (Execution plan §3.4, Stage 1 MIL explained §4):
  1. Load per-video [n_clips, 2048] fp16 features as float32.
  2. L2-normalise each clip's 2048-d feature vector.
  3. Pool training videos to 32 equal consecutive segments (mean for n >= 32,
     nearest-neighbour repetition for n < 32).
  4. Parse frame-level test annotations from Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
  5. Build a deterministic stratified 10% hold-out split (37 anomaly + 40 normal on L1+L2).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_LANES = ("L1", "L2")
DEFAULT_FRAMES_PER_CLIP = 78  # 13 frames * stride 6 (~2.6 s at 30 fps)
DEFAULT_NUM_SEGMENTS = 32
DEFAULT_FEAT_DIM = 2048


@dataclass
class VideoRecord:
    video_id: str
    split: str            # "train" or "test"
    label: int            # 1 = anomaly, 0 = normal
    cls: str              # e.g. "Arson", "Burglary", ..., "Normal"
    lane: str             # "L1", "L2", "L2X"
    rel_path: str
    features: np.ndarray  # [n_clips, D] float32, L2-normalised per clip
    n_frames: int
    fps: float
    frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP
    intervals: tuple[tuple[int, int], ...] = ()
    frame_labels: np.ndarray | None = None  # [n_frames] int32 for test videos


def l2_normalise(feats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise each clip vector [..., D] in float32."""
    arr = np.asarray(feats, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, eps)


def pool_to_segments(feats: np.ndarray, num_segments: int = DEFAULT_NUM_SEGMENTS) -> np.ndarray:
    """Pool L2-normalised per-clip features [n_clips, D] into [num_segments, D].

    - If n_clips >= num_segments: split into `num_segments` consecutive equal-ish
      groups (np.array_split) and average each group.
    - If n_clips < num_segments: repeat clips consecutively so every segment gets
      its corresponding clip (e.g. 16 clips -> each clip reused for 2 segments).
    """
    arr = np.asarray(feats, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D features [n_clips, D], got {arr.shape}")
    n = arr.shape[0]
    if n == num_segments:
        return arr.copy()
    if n > num_segments:
        groups = np.array_split(arr, num_segments, axis=0)
        return np.stack([g.mean(axis=0) for g in groups], axis=0).astype(np.float32)
    idx = (np.arange(num_segments, dtype=np.int64) * n) // num_segments
    return arr[idx].astype(np.float32)


def load_manifest(
    manifest_path: str | Path,
    lanes: Sequence[str] = DEFAULT_LANES,
) -> list[dict[str, str]]:
    """Read ucf_subset.csv and keep only rows belonging to `lanes`."""
    lane_set = set(lanes)
    with open(manifest_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["lane"] in lane_set]
    rows.sort(key=lambda r: (r["split"], -int(r["label"]), r["video_id"]))
    return rows


def load_temporal_annotations(
    ann_path: str | Path,
) -> dict[str, tuple[str, tuple[tuple[int, int], ...]]]:
    """Parse Temporal_Anomaly_Annotation_for_Testing_Videos.txt.

    Each line has: `<video>.mp4  <Class>  <start1>  <end1>  <start2>  <end2>`
    where `-1` marks absent intervals. Returns `{video_id: (cls, ((s1, e1), ...))}`.
    """
    out: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {}
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            vid = parts[0]
            if vid.endswith(".mp4"):
                vid = vid[:-4]
            cls_name = parts[1]
            coords = [int(x) for x in parts[2:6]]
            ivs: list[tuple[int, int]] = []
            for s, e in ((coords[0], coords[1]), (coords[2], coords[3])):
                if s != -1 and e != -1 and e >= s:
                    ivs.append((s, e))
            out[vid] = (cls_name, tuple(ivs))
    return out


def build_frame_labels(
    intervals: Iterable[tuple[int, int]],
    n_frames: int,
) -> np.ndarray:
    """Build binary [n_frames] ground-truth array from 1-based inclusive UCF intervals."""
    labels = np.zeros(n_frames, dtype=np.int32)
    for start, end in intervals:
        lo = max(0, start - 1) if start >= 1 else 0
        hi = min(n_frames, end)
        if hi > lo:
            labels[lo:hi] = 1
    return labels


def load_video_feature(
    features_dir: str | Path,
    row: dict[str, str],
    annotations: dict[str, tuple[str, tuple[tuple[int, int], ...]]] | None = None,
    normalise: bool = True,
    expected_dim: int | None = None,
) -> VideoRecord:
    """Load one video's .npy features, .json sidecar, and (for test) frame labels."""
    features_dir = Path(features_dir)
    vid = row["video_id"]
    npy_path = features_dir / f"{vid}.npy"
    json_path = features_dir / f"{vid}.json"

    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(f"{npy_path}: invalid shape {arr.shape}")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(f"{npy_path}: expected dim {expected_dim}, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{npy_path}: contains NaN or Inf")

    feats = l2_normalise(arr) if normalise else arr.astype(np.float32)

    fps = 30.0
    frames_per_clip = DEFAULT_FRAMES_PER_CLIP
    n_frames = int(feats.shape[0]) * frames_per_clip
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        fps = float(meta.get("fps") or 30.0)
        stride = int(meta.get("stride") or 6)
        clip_len = int(meta.get("clip_len") or 13)
        frames_per_clip = stride * clip_len
        n_frames = int(meta.get("n_frames") or (feats.shape[0] * frames_per_clip))

    split = row["split"]
    label = int(row["label"])
    intervals: tuple[tuple[int, int], ...] = ()
    frame_labels: np.ndarray | None = None
    if split == "test":
        if annotations is not None and vid in annotations:
            intervals = annotations[vid][1]
        elif label == 1:
            raise KeyError(f"Test anomaly video {vid!r} missing from temporal annotations")
        frame_labels = build_frame_labels(intervals, n_frames)

    return VideoRecord(
        video_id=vid,
        split=split,
        label=label,
        cls=row["cls"],
        lane=row["lane"],
        rel_path=row.get("rel_path", ""),
        features=feats,
        n_frames=n_frames,
        fps=fps,
        frames_per_clip=frames_per_clip,
        intervals=intervals,
        frame_labels=frame_labels,
    )


def _allocate_counts(class_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    """Largest-remainder (Hamilton) allocation summing to `target_total`."""
    total = sum(class_sizes.values())
    if target_total <= 0 or total == 0:
        return {c: 0 for c in class_sizes}
    target_total = min(target_total, total)
    exact = {c: (sz * target_total) / total for c, sz in class_sizes.items()}
    alloc = {c: min(class_sizes[c], int(exact[c])) for c in class_sizes}
    rem = target_total - sum(alloc.values())
    if rem > 0:
        order = sorted(
            class_sizes.keys(),
            key=lambda c: (-(exact[c] - alloc[c]), c),
        )
        for c in order:
            if rem == 0:
                break
            if alloc[c] < class_sizes[c]:
                alloc[c] += 1
                rem -= 1
    return alloc


def stratified_holdout_split(
    records: Sequence[VideoRecord],
    val_frac: float = 0.10,
    seed: int = 415,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Deterministic stratified split by binary label and class.

    On UCF-Crime L1+L2 training set (374 anomaly + 400 normal) with val_frac=0.10,
    this produces exactly 37 anomaly + 40 normal = 77 hold-out videos and
    337 anomaly + 360 normal = 697 train-fit videos.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[VideoRecord]] = {0: [], 1: []}
    for r in records:
        by_label[r.label].append(r)

    train_fit: list[VideoRecord] = []
    holdout: list[VideoRecord] = []

    for label in (1, 0):
        group = by_label[label]
        if not group:
            continue
        target_val = max(1, int(round(len(group) * val_frac))) if len(group) >= 2 else 0
        by_cls: dict[str, list[VideoRecord]] = collections.defaultdict(list)
        for r in group:
            by_cls[r.cls].append(r)
        alloc = _allocate_counts({c: len(v) for c, v in by_cls.items()}, target_val)
        for cls_name in sorted(by_cls.keys()):
            cls_recs = sorted(by_cls[cls_name], key=lambda r: r.video_id)
            rng.shuffle(cls_recs)
            k = alloc[cls_name]
            holdout.extend(cls_recs[:k])
            train_fit.extend(cls_recs[k:])

    train_fit.sort(key=lambda r: (-r.label, r.video_id))
    holdout.sort(key=lambda r: (-r.label, r.video_id))
    return train_fit, holdout


def load_ucf_splits(
    features_dir: str | Path,
    manifest_path: str | Path,
    annotations_path: str | Path,
    lanes: Sequence[str] = DEFAULT_LANES,
    val_frac: float = 0.10,
    split_seed: int = 415,
    expected_dim: int | None = DEFAULT_FEAT_DIM,
    skip_missing: bool = False,
) -> tuple[list[VideoRecord], list[VideoRecord], list[VideoRecord]]:
    """Load (train_fit, holdout, test) VideoRecord lists."""
    rows = load_manifest(manifest_path, lanes=lanes)
    annotations = load_temporal_annotations(annotations_path)
    features_dir = Path(features_dir)

    train_all: list[VideoRecord] = []
    test_all: list[VideoRecord] = []
    skipped: list[str] = []

    for row in rows:
        npy_path = features_dir / f"{row['video_id']}.npy"
        if not npy_path.exists():
            if skip_missing:
                skipped.append(row["video_id"])
                continue
            raise FileNotFoundError(f"Missing feature file: {npy_path}")
        rec = load_video_feature(
            features_dir=features_dir,
            row=row,
            annotations=annotations,
            normalise=True,
            expected_dim=expected_dim,
        )
        if rec.split == "train":
            train_all.append(rec)
        else:
            test_all.append(rec)

    if skipped:
        print(f"[warn] skipped {len(skipped)} missing feature files", flush=True)

    train_fit, holdout = stratified_holdout_split(train_all, val_frac=val_frac, seed=split_seed)
    return train_fit, holdout, test_all


def validate_feature_dir(
    features_dir: str | Path,
    manifest_path: str | Path,
    lanes: Sequence[str] = DEFAULT_LANES,
    expected_dim: int = DEFAULT_FEAT_DIM,
) -> dict[str, object]:
    """Verify every manifest video in `lanes` has a valid finite [n, expected_dim] .npy."""
    features_dir = Path(features_dir)
    rows = load_manifest(manifest_path, lanes=lanes)
    missing: list[str] = []
    missing_json: list[str] = []
    bad_shape: list[tuple[str, tuple[int, ...]]] = []
    non_finite: list[str] = []
    load_errors: list[tuple[str, str]] = []
    clip_counts: list[int] = []
    by_lane = collections.Counter()
    ok_by_lane = collections.Counter()

    for r in rows:
        vid = r["video_id"]
        lane = r["lane"]
        by_lane[lane] += 1
        npy_path = features_dir / f"{vid}.npy"
        json_path = features_dir / f"{vid}.json"
        if not npy_path.exists():
            missing.append(vid)
            continue
        if not json_path.exists():
            missing_json.append(vid)
        try:
            arr = np.load(npy_path)
        except Exception as exc:
            load_errors.append((vid, f"{type(exc).__name__}: {exc}"))
            continue
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != expected_dim:
            bad_shape.append((vid, tuple(int(x) for x in arr.shape)))
            continue
        if not np.isfinite(arr).all():
            non_finite.append(vid)
            continue
        clip_counts.append(int(arr.shape[0]))
        ok_by_lane[lane] += 1

    return {
        "total_expected": len(rows),
        "ok": len(clip_counts),
        "by_lane": {lane: f"{ok_by_lane[lane]}/{by_lane[lane]}" for lane in sorted(by_lane)},
        "missing": missing,
        "missing_json": missing_json,
        "bad_shape": bad_shape,
        "non_finite": non_finite,
        "load_errors": load_errors,
        "min_clips": int(min(clip_counts)) if clip_counts else 0,
        "median_clips": float(np.median(clip_counts)) if clip_counts else 0.0,
        "max_clips": int(max(clip_counts)) if clip_counts else 0,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Validate UCF-Crime X3D-S feature directory.")
    ap.add_argument("--features", required=True, help="Directory with <video_id>.npy and .json")
    ap.add_argument("--manifest", default="data/manifests/ucf_subset.csv")
    ap.add_argument("--lanes", default="L1,L2")
    ap.add_argument("--dim", type=int, default=DEFAULT_FEAT_DIM)
    args = ap.parse_args()

    report = validate_feature_dir(
        features_dir=args.features,
        manifest_path=args.manifest,
        lanes=[x.strip() for x in args.lanes.split(",") if x.strip()],
        expected_dim=args.dim,
    )
    print(json.dumps(report, indent=2))
    has_issues = any(
        report[k] for k in ("missing", "bad_shape", "non_finite", "load_errors")
    )
    raise SystemExit(1 if has_issues else 0)


if __name__ == "__main__":
    _main()
