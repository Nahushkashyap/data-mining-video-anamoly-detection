#!/usr/bin/env python3
"""Download every *pretrained* model the app needs into models_cache/ (one time, about 2 GB).

Run from the repository root, inside the virtual environment:

    python scripts/fetch_models.py            # everything
    python scripts/fetch_models.py clip x3d   # only some of them

What it fetches (all frozen; none of these is trained by us):
    x3d          X3D-S, Kinetics-400 (torch.hub facebookresearch/pytorchvideo)   -> models_cache/torch
    clip         CLIP ViT-B/16 (openai/clip-vit-base-patch16)                    -> models_cache/huggingface
    clap         MS-CLAP 2023 weights + GPT-2 text encoder                       -> models_cache/huggingface
    efficientat  EfficientAT mn10_as code + AudioSet weights (mAP 0.471)         -> models_cache/efficientat

The two checkpoints trained in this project are NOT downloaded here (see README, "Trained checkpoints"):
    checkpoints/best_seed416.pt                    video Stage 1 MIL head
    models_cache/efficientat/home_head_v1.pt       home-security audio head (optional; without it the
                                                   app falls back to the zero-shot AudioSet trigger)
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "models_cache"
os.environ["TORCH_HOME"] = str(MC / "torch")
os.environ["HF_HOME"] = str(MC / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "0"  # the app itself runs offline; this script is the one place that downloads
sys.path.insert(0, str(ROOT))

EAT_REPO_ZIP = "https://github.com/fschmid56/EfficientAT/archive/refs/heads/main.zip"
EAT_WEIGHTS_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt"


def fetch_x3d() -> None:
    import torch

    torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True, trust_repo=True)


def fetch_clip() -> None:
    from transformers import CLIPModel, CLIPProcessor

    from vad.stage2.clip_scorer import DEFAULT_CLIP_MODEL

    CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL)
    CLIPProcessor.from_pretrained(DEFAULT_CLIP_MODEL)


def fetch_clap() -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    hf_hub_download("microsoft/msclap", "CLAP_weights_2023.pth")
    AutoTokenizer.from_pretrained("gpt2")  # msclap's text encoder (config_2023.yml: text_model 'gpt2')
    AutoModel.from_pretrained("gpt2")


def fetch_efficientat() -> None:
    dst = MC / "efficientat"
    src = dst / "src"
    dst.mkdir(parents=True, exist_ok=True)
    if not (src / "models" / "mn").is_dir():
        print("  downloading EfficientAT source (models/, helpers/, metadata/)")
        data = urllib.request.urlopen(EAT_REPO_ZIP, timeout=120).read()
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
            top = next(Path(tmp).iterdir())
            src.mkdir(parents=True, exist_ok=True)
            for name in ("models", "helpers", "metadata"):
                shutil.copytree(top / name, src / name, dirs_exist_ok=True)
            if (top / "LICENSE").exists():
                shutil.copy2(top / "LICENSE", src / "LICENSE")
    weights = dst / "mn10_as_mAP_471.pt"
    if not weights.exists():
        print("  downloading mn10_as_mAP_471.pt")
        urllib.request.urlretrieve(EAT_WEIGHTS_URL, weights)


STEPS = {"x3d": fetch_x3d, "clip": fetch_clip, "clap": fetch_clap, "efficientat": fetch_efficientat}


def main(argv: list[str]) -> int:
    names = argv or list(STEPS)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        print(f"unknown model(s): {unknown}; choose from {list(STEPS)}")
        return 2
    for name in names:
        print(f"[fetch] {name} ...", flush=True)
        STEPS[name]()
        print(f"[fetch] {name}: ok", flush=True)
    print("All requested models are in models_cache/.")
    for p in (ROOT / "checkpoints" / "best_seed416.pt", MC / "efficientat" / "home_head_v1.pt"):
        if not p.exists():
            print(f"note: trained checkpoint missing: {p.relative_to(ROOT)} (see README)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
