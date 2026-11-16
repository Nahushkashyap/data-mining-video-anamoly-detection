"""Audio loading for the audio branch: decode any video/audio file to mono float32 PCM via ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SILENCE_DBFS = -60.0  # windows quieter than this are treated as digital silence


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH (macOS: brew install ffmpeg; Colab has it preinstalled)")
    return exe


def load_audio(path: str | Path, sr: int) -> np.ndarray | None:
    """Return mono float32 PCM in [-1, 1] at `sr` Hz, or None if the file has no audio stream."""
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        err = p.stderr.decode(errors="replace")
        if "does not contain any stream" in err or "Output file #0 does not contain" in err or "matches no streams" in err:
            return None
        raise RuntimeError(f"ffmpeg failed on {path}: {err.strip()[:300]}")
    if not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    import torch
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), sr_in, sr_out).numpy()


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(r, 1e-6))
