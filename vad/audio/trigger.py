"""Audio Stage 1: cheap AudioSet trigger (EfficientAT mn10_as, mAP 0.471, 4.9 M params).

Implements recommendation P2 of *Audio visual video anomaly detection.md*: score 1 s audio
windows for the acoustically salient AudioSet classes (gunshot, machine gun, explosion,
screaming, shout/yell, glass shatter, fire alarm) and gate on the max probability. The
trigger is zero-shot (pretrained on AudioSet, nothing trained on our data).

Weights + model code live in `models_cache/efficientat/` (project-local).
"""
from __future__ import annotations

import contextlib
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vad.audio.audio_io import SILENCE_DBFS, rms_dbfs
from vad.audio.home_head import GATE, HOME_CLASSES, HomeHead, head_enabled

ROOT = Path(__file__).resolve().parents[2]
EAT_DIR = ROOT / "models_cache" / "efficientat"
EAT_WEIGHTS = EAT_DIR / "mn10_as_mAP_471.pt"
EAT_SR = 32000
WINDOW_SEC = 1.0
TILE_SEC = 10.0  # see AudioTrigger.probs
# Calibrated 20 Nov on the SESA *train* split (never test): lowest τ_a whose full two-stage branch keeps the
# casual-sound false-alarm rate <= 5% is 0.24; 0.25 also meets it (train FAR 0.050). SESA test at 0.25:
# threat recall 0.461, correct class 0.421, casual FAR 0.0 (data/audio_eval/results/sesa-calibrate_eval.json).
ZEROSHOT_AUDIO_TAU = 0.25
# With the trained home-security head (default since 22 Nov) scores are calibrated so each class's own
# val-chosen threshold sits at GATE = 0.5; VAD_AUDIO_MODEL=zeroshot restores the original trigger and 0.25.
USE_HEAD = head_enabled()
DEFAULT_AUDIO_TAU = GATE if USE_HEAD else ZEROSHOT_AUDIO_TAU

# trigger group -> AudioSet display names (P2 list + close synonyms). Groups map to Stage 2 classes.
TRIGGER_GROUPS: dict[str, list[str]] = {
    "shooting": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun"],
    "explosion": ["Explosion", "Boom"],
    "screaming": ["Screaming", "Shout", "Yell", "Battle cry", "Children shouting"],
    "glass_break": ["Glass", "Shatter", "Breaking"],
    "alarm": ["Fire alarm", "Smoke detector, smoke alarm", "Siren", "Civil defense siren", "Alarm"],
}
# zero-shot AudioSet labels per *home-head* class: used only as the baseline when evaluating the trained head
HOME_ZEROSHOT_GROUPS: dict[str, list[str]] = {
    "scream": TRIGGER_GROUPS["screaming"],
    "glass_break": TRIGGER_GROUPS["glass_break"],
    "loud_bang": ["Bang", "Slam", "Thump, thud", "Smash, crash", "Knock"],
    "shooting": TRIGGER_GROUPS["shooting"],
    "explosion": TRIGGER_GROUPS["explosion"],
    "alarm": TRIGGER_GROUPS["alarm"],
}
# hard negatives reported for context only (never trigger): fireworks, pops, door slams
HARD_NEGATIVES = ["Fireworks", "Firecracker", "Burst, pop", "Slam", "Door"]


@dataclass(frozen=True)
class TriggerResult:
    times: np.ndarray            # [W] window start seconds
    scores: np.ndarray           # [W] max trigger probability (0 for silent windows)
    group_scores: dict           # group -> [W]
    silent: np.ndarray           # [W] bool, RMS < SILENCE_DBFS
    dbfs: np.ndarray             # [W]
    top_group: list              # [W] argmax group name per window
    hard_neg: np.ndarray         # [W] max hard-negative probability
    feats: np.ndarray | None = None  # [W, 960] EfficientAT embeddings (what the trained head sees)


def _load_labels() -> list[str]:
    with open(EAT_DIR / "src" / "metadata" / "class_labels_indices.csv", encoding="utf-8") as f:
        return [r["display_name"] for r in csv.DictReader(f)]


class AudioTrigger:
    def __init__(self, device: str = "cpu", use_head: bool | None = None) -> None:
        src = str(EAT_DIR / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        with contextlib.chdir(src):  # helpers/utils.py opens metadata/*.csv relative to cwd at import
            from models.mn.model import get_model  # type: ignore  # EfficientAT code (MIT)
            from models.preprocess import AugmentMelSTFT  # type: ignore

        self.device = device
        with contextlib.redirect_stdout(io.StringIO()):  # get_model prints the whole network
            self.model = get_model(width_mult=1.0, pretrained_name=None, head_type="mlp", se_dims="c")
            self.mel = AugmentMelSTFT(n_mels=128, sr=EAT_SR, win_length=800, hopsize=320)
        state = torch.load(EAT_WEIGHTS, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"EfficientAT weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.model.to(device).eval()
        # mel front end: on CUDA it runs on the GPU (CPU STFT was the bottleneck on Colab's 2 vCPUs, GPU idle);
        # on MPS it stays on CPU (device mismatch in the pre-emphasis conv)
        self.mel_device = device if str(device).startswith("cuda") else "cpu"
        self.mel.to(self.mel_device).eval()
        labels = _load_labels()
        idx = {n: i for i, n in enumerate(labels)}
        self.labels = labels
        self.group_idx = {g: [idx[n] for n in names if n in idx] for g, names in TRIGGER_GROUPS.items()}
        self.neg_idx = [idx[n] for n in HARD_NEGATIVES if n in idx]
        self.home_zs_idx = {g: [idx[n] for n in names if n in idx] for g, names in HOME_ZEROSHOT_GROUPS.items()}
        missing_names = [n for names in TRIGGER_GROUPS.values() for n in names if n not in idx]
        if missing_names:
            print(f"[audio-trigger] AudioSet labels not found (ignored): {missing_names}", flush=True)
        self.head = HomeHead(device=device) if (USE_HEAD if use_head is None else use_head) else None
        self.mode = "home_head_v1" if self.head is not None else "zeroshot"

    def probs(self, chunks: np.ndarray, batch: int = 16) -> np.ndarray:
        return self.probs_and_features(chunks, batch)[0]

    @torch.inference_mode()
    def probs_and_features(self, chunks: np.ndarray, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """chunks [W, n] float32 at 32 kHz -> AudioSet sigmoid probabilities [W, 527].

        mn10 was trained on 10 s clips and saturates on inputs shorter than ~3 s (unrelated
        classes hit p=1.0 on 1 s speech). Each window is therefore tiled to TILE_SEC before
        scoring, which restores sane outputs (checked 20 Nov: speech->Speech 0.63,
        beeping 3.1 kHz tone->Buzzer/Beep/Fire alarm, white noise->White noise).

        Also returns the pooled mn10 embedding of each window ([W, 960]), the input of the trained head.
        """
        reps = max(1, int(round(TILE_SEC * EAT_SR / chunks.shape[1])))
        out, feats = [], []
        for i in range(0, len(chunks), batch):
            x = torch.from_numpy(np.ascontiguousarray(np.tile(chunks[i:i + batch], (1, reps)), dtype=np.float32))
            mel = self.mel(x.to(self.mel_device)).to(self.device)
            logits, f = self.model(mel.unsqueeze(1))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
            feats.append(f.float().reshape(len(x), -1).cpu().numpy())
        if not out:
            return np.zeros((0, len(self.labels)), np.float32), np.zeros((0, 960), np.float32)
        return np.concatenate(out), np.concatenate(feats)

    def score(self, wav32k: np.ndarray, window_sec: float = WINDOW_SEC, hop_sec: float | None = None) -> TriggerResult:
        """Score `window_sec` windows of a 32 kHz mono waveform (non-overlapping unless `hop_sec` is set)."""
        n = int(EAT_SR * window_sec)
        h = int(EAT_SR * (hop_sec or window_sec))
        W = max(1, int(np.ceil(max(0, len(wav32k) - n) / h)) + 1)
        padded = np.zeros((W - 1) * h + n, dtype=np.float32)
        padded[: len(wav32k)] = wav32k[: len(padded)]
        chunks = np.stack([padded[i * h: i * h + n] for i in range(W)])
        dbfs = np.array([rms_dbfs(c) for c in chunks], dtype=np.float32)
        silent = dbfs < SILENCE_DBFS
        p, feats = self.probs_and_features(chunks)
        if self.head is not None:
            cal = self.head.calibrated(feats)
            groups = {c: cal[:, k] for k, c in enumerate(self.head.classes)}
        else:
            groups = {g: (p[:, ix].max(axis=1) if ix else np.zeros(W, np.float32)) for g, ix in self.group_idx.items()}
        stack = np.stack(list(groups.values()), axis=1)  # [W, G]
        scores = stack.max(axis=1)
        scores[silent] = 0.0
        names = list(groups.keys())
        top = [names[int(i)] for i in stack.argmax(axis=1)]
        hard = p[:, self.neg_idx].max(axis=1) if self.neg_idx else np.zeros(W, np.float32)
        return TriggerResult(
            times=np.arange(W, dtype=np.float32) * (hop_sec or window_sec),
            scores=scores.astype(np.float32),
            group_scores={g: v.astype(np.float32) for g, v in groups.items()},
            silent=silent,
            dbfs=dbfs,
            top_group=top,
            hard_neg=hard.astype(np.float32),
            feats=feats,
        )
