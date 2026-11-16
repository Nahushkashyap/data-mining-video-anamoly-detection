"""XD-Violence dataset loader for paired I3D RGB (1024-d) and VGGish audio (128-d) features.

Supports:
  - Normalising Kaggle `#` -> `_` filename differences so I3D, VGGish, and
    official annotations.txt align 1-to-1 across all 3,954 train and 800 test videos.
  - Binary video labels from filename (`_label_A` -> 0 normal, else 1 violent).
  - Frame-level test labels from annotations.txt (16 frames per I3D/VGGish snippet).
  - Per-snippet L2 normalisation and deterministic stratified 10% hold-out split.
"""
from __future__ import annotations

import collections
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from vad.data.ucf_dataset import l2_normalise

XDV_FRAMES_PER_SNIPPET = 16


@dataclass
class XDVVideoRecord:
    video_id: str          # canonical normalised stem (without __0 / __vggish)
    split: str             # "train" or "test"
    label: int             # 0 = normal (_label_A), 1 = violent
    cls_tag: str           # e.g. "A", "B1-0-0", "G-B2-B6"
    video_features: np.ndarray | None   # [T, D_v] float32, L2-normalised
    audio_features: np.ndarray | None   # [T, D_a] float32, L2-normalised
    n_snippets: int
    n_frames: int
    intervals: tuple[tuple[int, int], ...] = ()
    frame_labels: np.ndarray | None = None


def canonical_xdv_id(name: str) -> str:
    """Canonicalize an XD-Violence video ID across Kaggle (`___`) and official (`__#`) names."""
    s = Path(name).name
    for suf in (".npy", ".mp4"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"__vggish$", "", s)
    s = re.sub(r"__[0-4]$", "", s)
    # Kaggle replaces `#` with `_` (so `__#` becomes `___`)
    s = s.replace("#", "_")
    return s


def parse_xdv_label(video_id: str) -> tuple[int, str]:
    """Return (binary_label, class_tag) from XD-Violence filename suffix `_label_<TAG>`."""
    m = re.search(r"_label_([A-Za-z0-9\-]+)$", video_id)
    if not m:
        raise ValueError(f"Cannot parse XD-Violence label from {video_id!r}")
    tag = m.group(1)
    return (0 if tag == "A" else 1), tag


def load_xdv_annotations(ann_path: str | Path) -> dict[str, tuple[tuple[int, int], ...]]:
    """Parse XD-Violence annotations.txt (`<video_id> s1 e1 [s2 e2 ...]`)."""
    out: dict[str, tuple[tuple[int, int], ...]] = {}
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            vid = canonical_xdv_id(parts[0])
            nums = [int(x) for x in parts[1:]]
            ivs = tuple((nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2) if nums[i + 1] >= nums[i])
            out[vid] = ivs
    return out


def build_xdv_frame_labels(intervals: Iterable[tuple[int, int]], n_frames: int) -> np.ndarray:
    """Build [n_frames] binary frame ground truth from XD-Violence 0-based [s, e] intervals."""
    gt = np.zeros(n_frames, dtype=np.int32)
    for s, e in intervals:
        lo = max(0, int(s))
        hi = min(n_frames, int(e) + 1)
        if hi > lo:
            gt[lo:hi] = 1
    return gt


def _index_npy_dir(directory: Path | None) -> dict[str, Path]:
    if directory is None or not directory.exists():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(directory.glob("*.npy")):
        out[canonical_xdv_id(p.name)] = p
    return out


def _align_modalities(
    v_arr: np.ndarray | None,
    a_arr: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Ensure video and audio snippet sequences have matching length T."""
    if v_arr is not None and a_arr is not None:
        t = min(v_arr.shape[0], a_arr.shape[0])
        if t < 1:
            raise ValueError("Empty feature sequence after modality alignment")
        return v_arr[:t], a_arr[:t], int(t)
    if v_arr is not None:
        return v_arr, None, int(v_arr.shape[0])
    if a_arr is not None:
        return None, a_arr, int(a_arr.shape[0])
    raise ValueError("Both video and audio features are None")


def _stratified_split(
    records: Sequence[XDVVideoRecord],
    val_frac: float = 0.10,
    seed: int = 415,
) -> tuple[list[XDVVideoRecord], list[XDVVideoRecord]]:
    rng = random.Random(seed)
    by_label: dict[int, list[XDVVideoRecord]] = collections.defaultdict(list)
    for r in records:
        by_label[r.label].append(r)

    train_fit: list[XDVVideoRecord] = []
    holdout: list[XDVVideoRecord] = []
    for label in (1, 0):
        grp = sorted(by_label[label], key=lambda r: r.video_id)
        if not grp:
            continue
        rng.shuffle(grp)
        k = max(1, int(round(len(grp) * val_frac))) if len(grp) >= 2 else 0
        holdout.extend(grp[:k])
        train_fit.extend(grp[k:])

    train_fit.sort(key=lambda r: (-r.label, r.video_id))
    holdout.sort(key=lambda r: (-r.label, r.video_id))
    return train_fit, holdout


def load_xdv_splits(
    xdv_root: str | Path,
    *,
    val_frac: float = 0.10,
    split_seed: int = 415,
    require_video: bool = True,
    require_audio: bool = True,
    frames_per_snippet: int = XDV_FRAMES_PER_SNIPPET,
) -> tuple[list[XDVVideoRecord], list[XDVVideoRecord], list[XDVVideoRecord]]:
    """Load paired (train_fit, holdout, test) XD-Violence records from `xdv_root`."""
    root = Path(xdv_root)
    ann_path = root / "annotations.txt"
    if not ann_path.exists():
        alt = root / "annotations_aligned.txt"
        if alt.exists():
            ann_path = alt
        else:
            raise FileNotFoundError(f"annotations.txt not found under {root}")
    annotations = load_xdv_annotations(ann_path)

    def _load_split(split: str) -> list[XDVVideoRecord]:
        v_map = _index_npy_dir(root / "i3d_rgb" / split) if require_video else {}
        a_map = _index_npy_dir(root / "vggish" / split) if require_audio else {}

        if require_video and require_audio:
            ids = sorted(set(v_map) & set(a_map))
            only_v = len(set(v_map) - set(a_map))
            only_a = len(set(a_map) - set(v_map))
            if only_v or only_a:
                print(f"[xdv:{split}] aligned {len(ids)} pairs (dropped {only_v} video-only, {only_a} audio-only)", flush=True)
        elif require_video:
            ids = sorted(v_map)
        else:
            ids = sorted(a_map)

        recs: list[XDVVideoRecord] = []
        for vid in ids:
            label, tag = parse_xdv_label(vid)
            v_arr = None
            a_arr = None
            if require_video:
                raw_v = np.load(v_map[vid])
                if raw_v.ndim != 2 or raw_v.shape[0] < 1 or not np.isfinite(raw_v).all():
                    continue
                v_arr = l2_normalise(raw_v)
            if require_audio:
                raw_a = np.load(a_map[vid])
                if raw_a.ndim != 2 or raw_a.shape[0] < 1 or not np.isfinite(raw_a).all():
                    continue
                a_arr = l2_normalise(raw_a)

            v_arr, a_arr, t = _align_modalities(v_arr, a_arr)
            n_frames = t * frames_per_snippet
            ivs: tuple[tuple[int, int], ...] = ()
            fl: np.ndarray | None = None
            if split == "test":
                ivs = annotations.get(vid, ())
                fl = build_xdv_frame_labels(ivs, n_frames)

            recs.append(
                XDVVideoRecord(
                    video_id=vid,
                    split=split,
                    label=label,
                    cls_tag=tag,
                    video_features=v_arr,
                    audio_features=a_arr,
                    n_snippets=t,
                    n_frames=n_frames,
                    intervals=ivs,
                    frame_labels=fl,
                )
            )
        return recs

    train_all = _load_split("train")
    test_all = _load_split("test")
    if not train_all or not test_all:
        raise RuntimeError(f"Failed to load XD-Violence splits from {root} (train={len(train_all)}, test={len(test_all)})")

    train_fit, holdout = _stratified_split(train_all, val_frac=val_frac, seed=split_seed)
    return train_fit, holdout, test_all
