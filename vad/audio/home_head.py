"""Trained audio Stage 1 head for home-security sounds (v1, trained 22 Nov 2026 on FSD50K).

The zero-shot AudioSet trigger (vad/audio/trigger.py) only reached glass-break F1 0.25 / gunshot F1 ~0.01 on
TUT Rare Sound Events and was never tested on screams or bangs. This head is trained on top of the *frozen*
EfficientAT mn10 embeddings of the same 1 s windows (tiled to 10 s) the app already computes, so it adds
<1 ms per window and needs no new model download at inference time.

Classes (multi-label, sigmoid):
    scream       Screaming, Shout, Yell
    glass_break  Shatter
    loud_bang    Slam, Thump_and_thud, Boom
    shooting     Gunshot_and_gunfire
    explosion    Explosion minus gunshots and fireworks
    alarm        Siren, Alarm minus telephone / ringtone / doorbell

Training: colab/job_audio_home.py -> vad.audio.train_home_head (MIL max-pooling over the windows of each
clip, because FSD50K labels are clip-level). Per-class thresholds are chosen on FSD50K *val*, never eval.

Weights: models_cache/efficientat/home_head_v1.pt. Set VAD_AUDIO_MODEL=zeroshot to fall back to the
original zero-shot trigger (kept as a backup).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HEAD_PATH = Path(os.environ.get("VAD_AUDIO_HEAD") or ROOT / "models_cache" / "efficientat" / "home_head_v1.pt")
HOME_CLASSES: tuple[str, ...] = ("scream", "glass_break", "loud_bang", "shooting", "explosion", "alarm")
# calibrated scores put every class's own val-chosen threshold at this value, so one τ_a slider works for all
GATE = 0.5


class HomeHeadNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = len(HOME_CLASSES), hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., D] -> [..., C] logits
        return self.net(x)


def calibrate(p: np.ndarray, thr: np.ndarray, gate: float = GATE) -> np.ndarray:
    """Monotone piecewise-linear map per class: [0, thr] -> [0, gate], [thr, 1] -> [gate, 1]."""
    thr = np.clip(thr, 1e-4, 1 - 1e-4)
    lo = p * (gate / thr)
    hi = gate + (p - thr) * ((1 - gate) / (1 - thr))
    return np.where(p <= thr, lo, hi).astype(np.float32)


class HomeHead:
    def __init__(self, path: str | Path = HEAD_PATH, device: str = "cpu") -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.classes: list[str] = list(ck["classes"])
        self.mean = np.asarray(ck["feat_mean"], np.float32)
        self.std = np.asarray(ck["feat_std"], np.float32)
        self.thresholds = np.asarray([ck["thresholds"][c] for c in self.classes], np.float32)
        self.net = HomeHeadNet(ck["in_dim"], len(self.classes), ck.get("hidden", 256))
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(device).eval()
        self.device = device
        self.meta: dict[str, Any] = {k: ck[k] for k in ("version", "trained", "seed", "val_map") if k in ck}

    @torch.inference_mode()
    def probs(self, feats: np.ndarray) -> np.ndarray:
        """feats [W, D] (EfficientAT pooled embeddings) -> raw sigmoid probabilities [W, C]."""
        x = torch.from_numpy(((feats - self.mean) / self.std).astype(np.float32)).to(self.device)
        return torch.sigmoid(self.net(x)).float().cpu().numpy()

    def calibrated(self, feats: np.ndarray) -> np.ndarray:
        return calibrate(self.probs(feats), self.thresholds)


def head_enabled() -> bool:
    return os.environ.get("VAD_AUDIO_MODEL", "home").lower() != "zeroshot" and HEAD_PATH.exists()
