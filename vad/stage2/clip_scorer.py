"""Stage 2 zero-shot CLIP anomaly classifier refactored from web_scraping/clip_summarize.py.

Implements Execution plan §3.4:
  - Model: CLIP ViT-B/16 (`openai/clip-vit-base-patch16`; loaded lazily on instantiation).
  - 9 classes: burglary, fire, shooting, fighting, robbery, explosion,
    masked_person, normal, unrelated.
  - Prompt embeddings per class are L2-normalised, averaged, and re-normalised.
  - Frame probabilities: softmax(100.0 * img @ text_emb.T), averaged over frames.
  - Output per window: top-3 classes, peak frame (highest non-normal probability),
    one-line summary, and suppression flag if P(normal) + P(unrelated) >= 0.6.
  - Real-vs-synthetic check is off by default for UCF-Crime (enable for scraped video).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch16"
DEFAULT_SUPPRESS_THRESHOLD = 0.6

# 9 approved classes (Execution plan §3.4)
CLASS_PROMPTS: dict[str, list[str]] = {
    "burglary": [
        "a surveillance camera video of a burglar breaking into a house",
        "a CCTV video of a thief stealing from a store",
        "a person forcing a door open to rob a home",
    ],
    "fire": [
        "a video of a building on fire with flames and smoke",
        "a house fire with fire fighters",
        "a CCTV video of a fire breaking out",
    ],
    "shooting": [
        "a surveillance video of a person firing a gun",
        "a shooting incident with people running away",
        "a person pointing a gun at someone",
    ],
    "fighting": [
        "a CCTV video of people fighting and punching each other",
        "a physical altercation between two people",
        "a street brawl caught on camera",
    ],
    "robbery": [
        "a CCTV video of a robbery in a store or gas station",
        "a surveillance video of a robber holding up a cashier",
        "a person robbing a shop or stealing money from a register",
    ],
    "explosion": [
        "a video of a large explosion with a fireball",
        "an explosion and smoke cloud caught on camera",
    ],
    "masked_person": [
        "a person wearing a mask hiding their face",
        "a masked intruder in a balaclava",
        "a person with a covered face on a security camera",
    ],
    "normal": [
        "real CCTV footage of a quiet street with nothing unusual happening",
        "a real security camera view of people walking calmly",
        "real surveillance footage of an empty room, shop or parking lot",
    ],
    "unrelated": [
        "a screenshot of a video game",
        "gameplay footage from a video game like GTA",
        "a cartoon or animation",
        "a scene from a movie or TV show",
        "a news presenter talking in a studio",
        "a person talking to the camera in a vlog",
        "a title card or text on a plain background",
        "a meme with large text",
    ],
}

STAGE2_CLASSES: tuple[str, ...] = tuple(CLASS_PROMPTS.keys())
NON_ANOMALY_CLASSES: frozenset[str] = frozenset({"normal", "unrelated"})
ANOMALY_CLASSES: tuple[str, ...] = tuple(c for c in STAGE2_CLASSES if c not in NON_ANOMALY_CLASSES)

# UCF-Crime label -> Stage 2 class name (Execution plan §3.4: arson counts as fire)
UCF_TO_STAGE2_CLASS: dict[str, str] = {
    "Arson": "fire",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Normal": "normal",
}

REALNESS_PROMPTS: dict[str, list[str]] = {
    "real": [
        "real CCTV security camera footage",
        "a real photo of a real place taken with a camera",
        "real smartphone video of a real event",
        "real news footage filmed on location",
        "a grainy real surveillance camera recording",
    ],
    "synthetic": CLASS_PROMPTS["unrelated"] + [
        "computer generated 3D graphics",
        "a video game screen with a minimap and on-screen HUD",
    ],
}


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_pil_list(frames: Sequence[Image.Image | np.ndarray | str | Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for f in frames:
        if isinstance(f, Image.Image):
            images.append(f.convert("RGB"))
        elif isinstance(f, np.ndarray):
            arr = f.astype(np.uint8) if f.dtype != np.uint8 else f
            images.append(Image.fromarray(arr).convert("RGB"))
        else:
            images.append(Image.open(f).convert("RGB"))
    if not images:
        raise ValueError("Expected at least one frame to score")
    return images


def summarize_window_probs(
    probs: np.ndarray | torch.Tensor,
    classes: Sequence[str] = STAGE2_CLASSES,
    *,
    frame_timestamps: Sequence[float] | None = None,
    suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
    top_k: int = 3,
    p_synthetic: np.ndarray | torch.Tensor | None = None,
    synthetic_threshold: float = 0.5,
) -> dict[str, Any]:
    """Convert per-frame class probabilities [F, C] into Stage 2 event fields.

    Pure post-processing helper (also testable without loading CLIP weights).
    """
    p = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != len(classes):
        raise ValueError(f"Expected probs shape [F, {len(classes)}], got {p.shape}")

    cls_list = list(classes)
    mean_probs = p.mean(axis=0)  # [C]
    order = np.argsort(-mean_probs)
    top_k_idx = order[:top_k].tolist()

    top3 = [
        {"class": cls_list[i], "prob": round(float(mean_probs[i]), 4)}
        for i in top_k_idx
    ]
    pred_class = cls_list[top_k_idx[0]]
    confidence = round(float(mean_probs[top_k_idx[0]]), 4)

    # Non-normal probability per frame = sum over anomaly classes (1 - P(normal) - P(unrelated))
    non_anom_idx = [i for i, c in enumerate(cls_list) if c in NON_ANOMALY_CLASSES]
    anom_idx = [i for i, c in enumerate(cls_list) if c not in NON_ANOMALY_CLASSES]

    non_anom_mean_prob = float(mean_probs[non_anom_idx].sum()) if non_anom_idx else 0.0
    frame_anom_prob = p[:, anom_idx].sum(axis=1) if anom_idx else np.zeros(p.shape[0], dtype=np.float32)
    peak_frame_idx = int(np.argmax(frame_anom_prob))
    peak_non_normal_prob = float(frame_anom_prob[peak_frame_idx])
    peak_cls_idx = anom_idx[int(np.argmax(p[peak_frame_idx, anom_idx]))] if anom_idx else top_k_idx[0]
    peak_cls = cls_list[peak_cls_idx]

    suppressed = non_anom_mean_prob >= suppress_threshold
    suppress_reason = "normal_or_unrelated" if suppressed else ""

    synthetic_median: float | None = None
    if p_synthetic is not None:
        ps = (
            p_synthetic.detach().cpu().numpy()
            if isinstance(p_synthetic, torch.Tensor)
            else np.asarray(p_synthetic, dtype=np.float32)
        )
        synthetic_median = round(float(np.median(ps)), 4)
        if synthetic_median >= synthetic_threshold:
            suppressed = True
            suppress_reason = "clip_synthetic"

    peak_ts = (
        round(float(frame_timestamps[peak_frame_idx]), 2)
        if frame_timestamps is not None and peak_frame_idx < len(frame_timestamps)
        else None
    )

    # Also compute anomaly-only ranking for reference / confusion matrix analysis
    anom_order = sorted(anom_idx, key=lambda i: -float(mean_probs[i]))
    top3_anomaly = [
        {"class": cls_list[i], "prob": round(float(mean_probs[i]), 4)}
        for i in anom_order[:top_k]
    ]

    ts_note = f" at {peak_ts:.2f}s" if peak_ts is not None else ""
    if suppressed:
        summary = (
            f"Suppressed ({suppress_reason}: P(normal)+P(unrelated)={non_anom_mean_prob:.0%}). "
            f"Top class: '{pred_class}' ({confidence:.0%})."
        )
    else:
        runners = ", ".join(f"{t['class']} {t['prob']:.0%}" for t in top3[1:])
        summary = (
            f"Detected '{pred_class}' ({confidence:.0%} avg over {p.shape[0]} frames; "
            f"peak non-normal frame #{peak_frame_idx}{ts_note}: '{peak_cls}', "
            f"P(anomaly)={peak_non_normal_prob:.0%})."
        )
        if runners:
            summary += f" Runner-up: {runners}."

    return {
        "class": pred_class,
        "confidence": confidence,
        "top3": top3,
        "top3_anomaly": top3_anomaly,
        "peak_frame_idx": peak_frame_idx,
        "peak_frame_ts": peak_ts,
        "peak_non_normal_prob": round(peak_non_normal_prob, 4),
        "non_anomaly_prob": round(non_anom_mean_prob, 4),
        "suppressed": suppressed,
        "suppress_reason": suppress_reason,
        "synthetic_score": synthetic_median,
        "summary": summary,
        "mean_probs": {cls_list[i]: round(float(mean_probs[i]), 4) for i in range(len(cls_list))},
    }


class ClipScorer:
    """Zero-shot CLIP ViT-B/16 scorer for Stage 2 flagged windows."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str = "auto",
        class_prompts: dict[str, list[str]] | None = None,
        use_realness: bool = False,
        logit_scale: float = 100.0,
    ) -> None:
        # Lazy transformers import so unit tests/mocking work even without transformers
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.device = pick_device(device)
        self.logit_scale = float(logit_scale)
        self.use_realness = bool(use_realness)
        self.class_prompts = class_prompts or CLASS_PROMPTS
        self.classes = list(self.class_prompts.keys())

        print(f"[clip] loading {self.model_name} on {self.device}", flush=True)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.text_emb = self._encode_prompts(self.class_prompts)  # [C, D]
        self.real_emb = self._encode_prompts(REALNESS_PROMPTS) if self.use_realness else None

    # Explicit encoder + projection calls are version-independent across transformers releases.
    def _text_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        return self.model.text_projection(out.pooler_output)

    def _image_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        return self.model.visual_projection(out.pooler_output)

    @torch.inference_mode()
    def _encode_prompts(self, prompt_sets: dict[str, list[str]]) -> torch.Tensor:
        embs = []
        for prompts in prompt_sets.values():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            e = self._text_features(inputs)
            e = torch.nn.functional.normalize(e, dim=-1).mean(dim=0)
            embs.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.stack(embs)  # [C, D]

    @torch.inference_mode()
    def _score_images(self, frames: Sequence[Image.Image | np.ndarray | str | Path]):
        """-> (L2-normalised image embeddings [F, D], cosine sims [F, C], probs [F, C], optional p_synthetic [F])."""
        images = _to_pil_list(frames)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        img = torch.nn.functional.normalize(self._image_features(inputs), dim=-1)
        sims = img @ self.text_emb.T
        probs = (self.logit_scale * sims).softmax(dim=-1).cpu().numpy()
        p_synth = None
        if self.real_emb is not None:
            p_synth = (self.logit_scale * (img @ self.real_emb.T)).softmax(dim=-1)[:, 1].cpu().numpy()
        return (img.float().cpu().numpy(), sims.float().cpu().numpy(), probs.astype(np.float32),
                (p_synth.astype(np.float32) if p_synth is not None else None))

    def score_frames(
        self,
        frames: Sequence[Image.Image | np.ndarray | str | Path],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Score RGB frames -> (probs [F, 9], optional p_synthetic [F])."""
        _, _, probs, p_synth = self._score_images(frames)
        return probs, p_synth

    def score_window(
        self,
        frames: Sequence[Image.Image | np.ndarray | str | Path],
        *,
        frame_timestamps: Sequence[float] | None = None,
        suppress_threshold: float = DEFAULT_SUPPRESS_THRESHOLD,
        top_k: int = 3,
        details: bool = False,
    ) -> dict[str, Any]:
        """Score a window's frames and return summarized Stage 2 prediction dict.

        details=True also returns the raw model output behind the summary (for the web UI's stage inspector):
        per-frame probabilities and cosine similarities to each class's prompt embedding, and the window's
        mean CLIP image embedding. The summary fields are identical either way.
        """
        emb, sims, probs, p_synth = self._score_images(frames)
        out = summarize_window_probs(
            probs=probs,
            classes=self.classes,
            frame_timestamps=frame_timestamps,
            suppress_threshold=suppress_threshold,
            top_k=top_k,
            p_synthetic=p_synth,
        )
        if details:
            mean_emb = emb.mean(axis=0)
            out["details"] = {
                "classes": list(self.classes),
                "frame_probs": np.round(probs, 4).tolist(),
                "frame_sims": np.round(sims, 4).tolist(),
                "logit_scale": self.logit_scale,
                "embedding": np.round(mean_emb / max(1e-8, float(np.linalg.norm(mean_emb))), 4).tolist(),
                "embedding_model": self.model_name,
            }
        return out
