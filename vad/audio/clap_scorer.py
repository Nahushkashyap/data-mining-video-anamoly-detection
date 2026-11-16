"""Audio Stage 2: zero-shot MS-CLAP 2023 verifier for windows flagged by the audio trigger.

Mirrors vad/stage2/clip_scorer.py: per-class prompt embeddings are L2-normalised, averaged and
re-normalised; softmax over CLAP's learned logit scale; suppress when P(normal)+P(unrelated) >= 0.6.
Hard negatives named in the research report (fireworks, door slams, balloon/bag pops) are prompts
of the `unrelated` class so they are suppressed rather than alerted.

Weights: `microsoft/msclap` CLAP_weights_2023.pth + `openai-community/gpt2` text encoder, both in
the project-local HF cache (models_cache/huggingface). The `msclap` package is installed with
--no-deps (its pins would downgrade numpy/transformers); only its model classes are used here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vad.audio.audio_io import resample
from vad.audio.home_head import head_enabled

CLAP_SR = 44100
CLAP_DURATION = 7  # seconds per CLAP input (2023 config)
TEXT_MODEL = "openai-community/gpt2"
DEFAULT_SUPPRESS_THRESHOLD = 0.6

# v0 = original zero-shot prompts (backup mode, VAD_AUDIO_MODEL=zeroshot)
AUDIO_CLASS_PROMPTS_V0: dict[str, list[str]] = {
    "shooting": ["gunshots", "a gun being fired", "rapid gunfire from a machine gun"],
    "explosion": ["an explosion", "a loud blast and rumble from an explosion", "a bomb exploding"],
    "fighting": ["people screaming and fighting", "a violent fight with shouting", "a person screaming in fear"],
    "glass_break": ["glass breaking", "a window being smashed", "shattering glass"],
    "fire_alarm": ["a fire alarm beeping", "a smoke alarm going off", "an emergency siren wailing"],
    "normal": ["quiet ambient background noise", "people talking calmly", "traffic noise on a street",
               "footsteps and room tone"],
    "unrelated": ["music playing", "a person narrating or speaking to the camera", "a crowd cheering at a sports event",
                  "fireworks popping", "a door slamming", "a balloon popping"],
}
# v1 (22 Nov, with the trained home-security head): adds `loud_bang` (door kicked in / banging, a burglary cue
# the user asked for) and therefore no longer lists "a door slamming" as an unrelated hard negative, which
# would veto every bang the head finds; a polite knock is a `normal` prompt instead.
AUDIO_CLASS_PROMPTS_V1: dict[str, list[str]] = {
    "shooting": ["gunshots", "a gun being fired", "rapid gunfire from a machine gun"],
    "explosion": ["an explosion", "a loud blast and rumble from an explosion", "a bomb exploding"],
    "fighting": ["people screaming and fighting", "a violent fight with shouting", "a person screaming in fear",
                 "a woman screaming for help"],
    "glass_break": ["glass breaking", "a window being smashed", "shattering glass"],
    "loud_bang": ["a loud bang", "someone banging hard on a door", "a door being kicked in",
                  "a heavy object crashing to the floor"],
    "fire_alarm": ["a fire alarm beeping", "a smoke alarm going off", "an emergency siren wailing"],
    "normal": ["quiet ambient background noise", "people talking calmly", "traffic noise on a street",
               "footsteps and room tone", "a gentle knock on a door"],
    "unrelated": ["music playing", "a person narrating or speaking to the camera", "a crowd cheering at a sports event",
                  "fireworks popping", "a balloon popping"],
}
AUDIO_CLASS_PROMPTS: dict[str, list[str]] = AUDIO_CLASS_PROMPTS_V1 if head_enabled() else AUDIO_CLASS_PROMPTS_V0
AUDIO_CLASSES: tuple[str, ...] = tuple(AUDIO_CLASS_PROMPTS)
AUDIO_NON_ANOMALY = frozenset({"normal", "unrelated"})
AUDIO_ANOMALY_CLASSES: tuple[str, ...] = tuple(c for c in AUDIO_CLASSES if c not in AUDIO_NON_ANOMALY)

# audio class -> video Stage 2 class (for fusion and UCF evaluation)
AUDIO_TO_VIDEO_CLASS: dict[str, str] = {
    "shooting": "shooting",
    "explosion": "explosion",
    "fighting": "fighting",
    "glass_break": "burglary",
    "loud_bang": "burglary",
    "fire_alarm": "fire",
}


def _hf_args() -> Any:
    import yaml
    from types import SimpleNamespace
    import msclap
    cfg = yaml.safe_load((Path(msclap.__file__).parent / "configs" / "config_2023.yml").read_text())
    cfg["text_model"] = TEXT_MODEL
    return SimpleNamespace(**cfg)


class ClapScorer:
    def __init__(self, device: str = "cpu", class_prompts: dict[str, list[str]] | None = None) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from msclap.models.clap import CLAP

        self.args = a = _hf_args()
        self.device = device
        self.class_prompts = class_prompts or AUDIO_CLASS_PROMPTS
        self.classes = list(self.class_prompts)
        model = CLAP(audioenc_name=a.audioenc_name, sample_rate=a.sampling_rate, window_size=a.window_size,
                     hop_size=a.hop_size, mel_bins=a.mel_bins, fmin=a.fmin, fmax=a.fmax, classes_num=a.num_classes,
                     out_emb=a.out_emb, text_model=a.text_model, transformer_embed_dim=a.transformer_embed_dim,
                     d_proj=a.d_proj)
        ckpt = hf_hub_download("microsoft/msclap", "CLAP_weights_2023.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        # transformers>=4.3x dropped GPT-2's causal-mask buffers ("attn.bias"/"masked_bias"); those are harmless.
        bad_missing = [k for k in missing if not k.endswith(("attn.bias", "attn.masked_bias"))]
        bad_unexp = [k for k in unexpected if not k.endswith(("attn.bias", "attn.masked_bias"))]
        if bad_missing or bad_unexp:
            raise RuntimeError(f"CLAP weights mismatch: missing={bad_missing[:6]} unexpected={bad_unexp[:6]}")
        self.model = model.to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(a.text_model)
        self.tok.add_special_tokens({"pad_token": "!"})
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.text_emb = self._encode_prompts(self.class_prompts)

    @torch.inference_mode()
    def _text(self, texts: Sequence[str]) -> torch.Tensor:
        enc = self.tok([t + " <|endoftext|>" for t in texts], max_length=self.args.text_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        x = {"input_ids": enc["input_ids"].to(self.device), "attention_mask": enc["attention_mask"].to(self.device)}
        return self.model.caption_encoder(x)

    def _encode_prompts(self, prompt_sets: dict[str, list[str]]) -> list[torch.Tensor]:
        """One [n_prompts, D] matrix per class.

        Unlike the CLIP stage, class scores use the MAX over a class's prompts, not the averaged
        prompt embedding: averaging the six diverse `unrelated` prompts blurred that class and
        let narrow classes win by default (TTS speech came out 'shooting' 0.59). With max-over-
        prompts: speech -> unrelated 1.0/0.93, beeping tone -> fire_alarm 1.0, synthetic bangs ->
        shooting 0.83 (checked 20 Nov). This is the CLIP-AD / WinCLIP point from the research report.
        """
        return [torch.nn.functional.normalize(self._text(p), dim=-1) for p in prompt_sets.values()]

    @staticmethod
    def crops(wav44k: np.ndarray, max_crops: int = 3) -> np.ndarray:
        """Fixed 7 s crops: repeat short audio (as msclap does); evenly spaced crops for long audio."""
        n = CLAP_SR * CLAP_DURATION
        if len(wav44k) == 0:
            wav44k = np.zeros(n, np.float32)
        if len(wav44k) <= n:
            reps = int(np.ceil(n / len(wav44k)))
            return np.tile(wav44k, reps)[:n][None].astype(np.float32)
        k = min(max_crops, int(np.ceil(len(wav44k) / n)))
        starts = np.linspace(0, len(wav44k) - n, k).astype(int)
        return np.stack([wav44k[s:s + n] for s in starts]).astype(np.float32)

    @torch.inference_mode()
    def _forward(self, wav44k: np.ndarray):
        """-> (mean probs [C], audio embeddings [K, D], cosine sims [K, C]) over the crops of a 44.1 kHz segment."""
        x = torch.from_numpy(self.crops(wav44k)).to(self.device)
        a = torch.nn.functional.normalize(self.model.audio_encoder(x)[0], dim=-1)  # [K, D]
        sims = torch.stack([(a @ t.T).max(dim=-1).values for t in self.text_emb], dim=-1)  # [K, C]
        p = (self.logit_scale * sims).softmax(dim=-1).mean(dim=0)
        return p.float().cpu().numpy(), a.float().cpu().numpy(), sims.float().cpu().numpy()

    def probs(self, wav44k: np.ndarray) -> np.ndarray:
        """Mean class probabilities [C] over the crops of a 44.1 kHz segment."""
        return self._forward(wav44k)[0]

    def score_segment(self, wav: np.ndarray, sr: int, suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
                      top_k: int = 3, details: bool = False) -> dict[str, Any]:
        p, emb, sims = self._forward(resample(wav, sr, CLAP_SR))
        order = np.argsort(-p)
        top = [{"class": self.classes[i], "prob": round(float(p[i]), 4)} for i in order[:top_k]]
        non = float(sum(p[i] for i, c in enumerate(self.classes) if c in AUDIO_NON_ANOMALY))
        anom = sorted([i for i, c in enumerate(self.classes) if c not in AUDIO_NON_ANOMALY], key=lambda i: -p[i])
        suppressed = non >= suppress_threshold
        return {
            "class": self.classes[order[0]],
            "confidence": round(float(p[order[0]]), 4),
            "top3": top,
            "top3_anomaly": [{"class": self.classes[i], "prob": round(float(p[i]), 4)} for i in anom[:top_k]],
            "non_anomaly_prob": round(non, 4),
            "suppressed": bool(suppressed),
            "mean_probs": {c: round(float(p[i]), 4) for i, c in enumerate(self.classes)},
            **({"details": {
                "classes": list(self.classes),
                "crop_sims": np.round(sims, 4).tolist(),  # [crops, C] max-over-prompts cosine similarity
                "n_crops": int(len(emb)),
                "logit_scale": round(self.logit_scale, 3),
                "embedding": np.round(emb.mean(0) / max(1e-8, float(np.linalg.norm(emb.mean(0)))), 4).tolist(),
                "embedding_model": "MS-CLAP 2023 audio encoder",
            }} if details else {}),
        }
