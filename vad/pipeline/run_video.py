"""End-to-end two-stage cascade: Stage 1 MIL gate -> Stage 2 CLIP -> JSON alert events.

Implements Execution plan §3.4:
  - Stage 1: score L2-normalised per-clip X3D-S features with MILHead + 3-clip moving average.
  - Windowing: select clips > τ; merge windows less than 2 clips apart; pad ±1 clip.
  - Frame sampling: 8 evenly spaced RGB frames per flagged window from the source video.
  - Stage 2: CLIP ViT-B/16 over 9 classes; suppress if P(normal) + P(unrelated) >= 0.6.
  - Output JSON events:
      {start_sec, end_sec, class, confidence, top3, peak_frame, summary}
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from vad.data.ucf_dataset import DEFAULT_FRAMES_PER_CLIP, l2_normalise
from vad.eval.metrics import score_clips
from vad.stage1.model import MILHead
from vad.stage2.clip_scorer import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_SUPPRESS_THRESHOLD,
    ClipScorer,
)


class WindowScorer(Protocol):
    def score_window(
        self,
        frames: Sequence[Image.Image | np.ndarray | str | Path],
        *,
        frame_timestamps: Sequence[float] | None = None,
        suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
        top_k: int = 3,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ClipWindow:
    start_clip: int   # inclusive, after ±1 padding
    end_clip: int     # inclusive, after ±1 padding
    raw_start: int    # inclusive, before padding
    raw_end: int      # inclusive, before padding
    peak_score: float
    start_frame: int
    end_frame: int    # exclusive
    start_sec: float
    end_sec: float


def extract_flagged_windows(
    clip_scores: np.ndarray,
    tau: float,
    *,
    n_frames: int,
    fps: float = 30.0,
    frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP,
    merge_gap_lt: int = 2,
    pad_clips: int = 1,
) -> list[ClipWindow]:
    """Turn smoothed per-clip scores into padded Stage 2 windows per §3.4.

    1. Find clips with score > tau.
    2. Group contiguous clips into [start, end] inclusive intervals.
    3. Merge consecutive windows when the gap between them is strictly less than
       `merge_gap_lt` clips (i.e. gap = next_start - prev_end - 1 < 2, so 0 or 1).
    4. Pad each merged window by ±`pad_clips` clamped to [0, n_clips - 1].
    """
    scores = np.asarray(clip_scores, dtype=np.float32).ravel()
    n_clips = int(scores.size)
    if n_clips == 0 or n_frames <= 0:
        return []
    fps = float(fps) if fps and fps > 0 else 30.0

    flagged = np.flatnonzero(scores > tau)
    if flagged.size == 0:
        return []

    # Step 2 + 3: build intervals and merge when gap < merge_gap_lt
    merged: list[list[int]] = [[int(flagged[0]), int(flagged[0])]]
    for idx in flagged[1:]:
        i = int(idx)
        gap = i - merged[-1][1] - 1
        if gap < merge_gap_lt:
            merged[-1][1] = i
        else:
            merged.append([i, i])

    # Step 4: pad ±1 clip and map to frame / second boundaries
    windows: list[ClipWindow] = []
    for raw_s, raw_e in merged:
        s = max(0, raw_s - pad_clips)
        e = min(n_clips - 1, raw_e + pad_clips)
        start_frame = min(n_frames - 1, s * frames_per_clip)
        end_frame = min(n_frames, (e + 1) * frames_per_clip)
        if end_frame <= start_frame:
            end_frame = min(n_frames, start_frame + 1)
        windows.append(
            ClipWindow(
                start_clip=s,
                end_clip=e,
                raw_start=raw_s,
                raw_end=raw_e,
                peak_score=float(scores[raw_s : raw_e + 1].max()),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                start_sec=round(start_frame / fps, 2),
                end_sec=round(end_frame / fps, 2),
            )
        )
    return windows


def sample_window_frames(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    num_frames: int = 8,
    fps: float | None = None,
) -> tuple[list[Image.Image], list[int], list[float]]:
    """Decode `num_frames` evenly spaced RGB frames from [start_frame, end_frame)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    try:
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        lo = max(0, int(start_frame))
        hi = int(end_frame)
        if total > 0:
            hi = min(hi, total)
        if hi <= lo:
            hi = lo + 1

        span = hi - lo
        # Centres of `num_frames` equal sub-intervals
        targets = [min(hi - 1, lo + int(span * (i + 0.5) / num_frames)) for i in range(num_frames)]

        images: list[Image.Image] = []
        frame_idxs: list[int] = []
        timestamps: list[float] = []
        for idx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
            frame_idxs.append(idx)
            timestamps.append(round(idx / vid_fps, 2))
    finally:
        cap.release()

    if not images:
        raise RuntimeError(f"Failed to decode any frames in [{start_frame}, {end_frame}) from {video_path}")
    return images, frame_idxs, timestamps


def load_stage1_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> tuple[MILHead, dict[str, Any]]:
    """Load trained Stage 1 MILHead and its saved metadata (including τ)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MILHead(
        in_dim=int(ckpt.get("in_dim", 2048)),
        hidden1=int(ckpt.get("hidden1", 512)),
        hidden2=int(ckpt.get("hidden2", 32)),
        dropout=float(ckpt.get("dropout", 0.6)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def run_video(
    video_path: str | Path,
    *,
    stage1_model: MILHead,
    scorer: WindowScorer,
    tau: float,
    device: torch.device | str = "cpu",
    features_npy: str | Path | None = None,
    features_json: str | Path | None = None,
    smooth_window: int = 3,
    frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP,
    num_stage2_frames: int = 8,
    suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
    peaks_dir: str | Path | None = None,
    include_suppressed: bool = False,
) -> dict[str, Any]:
    """Run the full Stage 1 -> gate -> Stage 2 cascade on a single video.

    If `features_npy` is provided, uses pre-extracted X3D-S clip features; otherwise
    extracts X3D-S features from `video_path` using `scripts/extract_x3d.py`.
    """
    t0 = time.time()
    video_path = Path(video_path)
    video_id = video_path.stem

    if features_npy is not None:
        raw_feats = np.load(features_npy)
        meta: dict[str, Any] = {}
        jpath = Path(features_json) if features_json is not None else Path(features_npy).with_suffix(".json")
        if jpath.exists():
            meta = json.loads(jpath.read_text(encoding="utf-8"))
        fps = float(meta.get("fps") or 30.0)
        stride = int(meta.get("stride") or 6)
        clip_len = int(meta.get("clip_len") or 13)
        frames_per_clip = stride * clip_len
        n_frames = int(meta.get("n_frames") or (raw_feats.shape[0] * frames_per_clip))
        stage1_extract_sec = 0.0
    else:
        from scripts.extract_x3d import CLIP_LEN, STRIDE, embed_video, load_x3d

        t_ext = time.time()
        dev_obj = torch.device(device)
        x3d, dtype = load_x3d(dev_obj)
        batch = {"cuda": 32, "mps": 16}.get(dev_obj.type, 8)
        raw_feats, n_frames, fps, _ = embed_video(x3d, dtype, dev_obj, str(video_path), batch=batch)
        frames_per_clip = CLIP_LEN * STRIDE
        stage1_extract_sec = round(time.time() - t_ext, 3)

    t_head = time.time()
    feats = l2_normalise(raw_feats)
    raw_scores, smoothed_scores = score_clips(
        stage1_model, feats, device=device, smooth_window=smooth_window
    )
    stage1_head_sec = round(time.time() - t_head, 4)

    windows = extract_flagged_windows(
        smoothed_scores,
        tau=tau,
        n_frames=n_frames,
        fps=fps,
        frames_per_clip=frames_per_clip,
    )

    peaks_path = Path(peaks_dir) if peaks_dir is not None else None
    if peaks_path is not None:
        peaks_path.mkdir(parents=True, exist_ok=True)

    t_s2 = time.time()
    events: list[dict[str, Any]] = []
    suppressed_events: list[dict[str, Any]] = []

    for w_idx, win in enumerate(windows):
        frames, frame_idxs, timestamps = sample_window_frames(
            video_path,
            start_frame=win.start_frame,
            end_frame=win.end_frame,
            num_frames=num_stage2_frames,
            fps=fps,
        )
        res = scorer.score_window(
            frames,
            frame_timestamps=timestamps,
            suppress_threshold=suppress_threshold,
            top_k=3,
        )
        p_idx = int(res["peak_frame_idx"])
        peak_raw_frame = frame_idxs[p_idx]
        peak_ts = timestamps[p_idx]

        if peaks_path is not None and not res["suppressed"]:
            out_img = peaks_path / f"{video_id}_event{w_idx:02d}_f{peak_raw_frame}.jpg"
            frames[p_idx].save(out_img, quality=90)
            peak_frame_ref = str(out_img)
        else:
            peak_frame_ref = f"frame_{peak_raw_frame}@{peak_ts:.2f}s"

        event = {
            "start_sec": win.start_sec,
            "end_sec": win.end_sec,
            "class": res["class"],
            "confidence": res["confidence"],
            "top3": res["top3"],
            "peak_frame": peak_frame_ref,
            "summary": res["summary"],
            "stage1_peak": round(win.peak_score, 4),
            "mean_probs": res["mean_probs"],  # all 9 CLIP class probabilities (used by audio fusion)
        }
        if res["suppressed"]:
            suppressed_events.append({**event, "suppress_reason": res["suppress_reason"]})
        else:
            events.append(event)

    stage2_sec = round(time.time() - t_s2, 3)
    flagged_clips = int(np.sum(smoothed_scores > tau))

    output: dict[str, Any] = {
        "video_id": video_id,
        "video_path": str(video_path),
        "fps": round(fps, 3),
        "n_frames": int(n_frames),
        "n_clips": int(len(smoothed_scores)),
        "tau": float(tau),
        "flagged_clips": flagged_clips,
        "escalated_clip_pct": round(100.0 * flagged_clips / max(1, len(smoothed_scores)), 2),
        "n_windows_flagged": len(windows),
        "n_events": len(events),
        "n_suppressed": len(suppressed_events),
        "timing_sec": {
            "stage1_extract": stage1_extract_sec,
            "stage1_head": stage1_head_sec,
            "stage2_clip": stage2_sec,
            "total": round(time.time() - t0, 3),
        },
        "events": events,
    }
    if include_suppressed:
        output["suppressed_events"] = suppressed_events
    return output


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run Stage 1 + Stage 2 anomaly detection on a video.")
    ap.add_argument("--video", required=True, help="Path to input .mp4 video")
    ap.add_argument("--checkpoint", required=True, help="Stage 1 MILHead checkpoint (.pt)")
    ap.add_argument("--features", default=None, help="Optional pre-extracted <video_id>.npy features")
    ap.add_argument("--tau", type=float, default=None, help="Override Stage 1 alert threshold τ (default: from checkpoint)")
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--use-realness", action="store_true", help="Enable synthetic/game realness check (off for UCF)")
    ap.add_argument("--suppress-threshold", type=float, default=DEFAULT_SUPPRESS_THRESHOLD)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--peaks-dir", default=None, help="Directory to save peak anomaly frame JPEGs")
    ap.add_argument("--out-json", default=None, help="Write JSON result to this file (else print to stdout)")
    ap.add_argument("--include-suppressed", action="store_true")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    args = ap.parse_args(argv)

    dev = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model, ckpt = load_stage1_checkpoint(args.checkpoint, device=dev)
    tau = args.tau if args.tau is not None else float(ckpt["tau_info"]["tau"])
    smooth_window = int(ckpt.get("smooth_window", 3))
    scorer = ClipScorer(
        model_name=args.clip_model,
        device=dev,
        use_realness=args.use_realness,
    )

    result = run_video(
        args.video,
        stage1_model=model,
        scorer=scorer,
        tau=tau,
        device=dev,
        features_npy=args.features,
        smooth_window=smooth_window,
        num_stage2_frames=args.num_frames,
        suppress_threshold=args.suppress_threshold,
        peaks_dir=args.peaks_dir,
        include_suppressed=args.include_suppressed,
    )
    payload = json.dumps(result, indent=2)
    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload + "\n", encoding="utf-8")
        print(f"[run_video] wrote {out_p} ({result['n_events']} events, {result['n_suppressed']} suppressed)", flush=True)
    else:
        print(payload)


if __name__ == "__main__":
    main()
