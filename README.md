# Two-Stage Audio-Visual Video Anomaly Detection for Real-Time Surveillance

**CS F415 Data Mining, BITS Pilani, First Semester 2026–27**
Karan Bhojani (2026H1120146P) · Nahush Kashyap M A (2026H1120142P) · Utkarsh Shendre (2026H1120136P)

A surveillance anomaly detector that runs about **10× faster than real time on a laptop** and reports
*what* happened, *when*, and *how confidently*, from both the video and its audio.

- **Video Stage 1 (cheap, every clip):** frozen **X3D-S** features for each 2.6 s clip → the **MIL ranking head** of
  our base paper (Sultani et al., CVPR 2018), trained from video-level labels only → smoothed score > τ.
- **Video Stage 2 (flagged windows only):** zero-shot **CLIP ViT-B/16** names the event (burglary, fire, shooting,
  fighting, robbery, explosion, masked person) or vetoes it as normal/unrelated.
- **Audio branch:** frozen **EfficientAT mn10** embeddings → our trained **home-security sound head** (scream, glass
  break, loud bang, gunshot, explosion, alarm) → zero-shot **MS-CLAP** verification → per-class late fusion with video.
  Audio-only alerts cover off-camera events.
- **Web app:** FastAPI + HTML/JS, runs every model locally, supports uploaded videos and a live camera + mic.
- **Web data mining:** YouTube/Instagram scraping and CLIP pre-labelling of surveillance-style videos.

📄 Full write-up: [`report/VAD_Final_Report.pdf`](report/VAD_Final_Report.pdf) (4 pages, IEEE style; LaTeX source alongside).

## Results (summary)

| What | Result |
|---|---|
| Stage 1, UCF-Crime (226-video test subset, 3 seeds) | **0.843 ± 0.014** frame ROC-AUC (base paper: 0.754 on the full test set) |
| CLIP veto (test, video level) | normal videos with an alert **29.3% → 22.0%**; anomalous videos detected 96.1% → 93.4% |
| CLIP class naming on true anomaly windows | top-1 34.1% (chance 14%), top-3 70.7% (chance 43%) |
| Audio + video fusion, XD-Violence | **0.681** frame AP vs 0.603 video only (+7.9 AP) |
| Trained home-sound head, unseen FSD50K clips | mAP **0.409 → 0.661 ± 0.003** vs zero-shot AudioSet scores |
| Speed (MacBook, PyTorch MPS) | X3D-S 80–120 ms / 2.6 s clip, MIL < 1 ms, CLIP ~130 ms per flagged window |

## Repository layout

```
vad/
  stage1/      model.py (MIL head + Sultani ranking loss), train.py (UCF-Crime), train_xdv.py (XD-Violence fusion)
  stage2/      clip_scorer.py (zero-shot CLIP, prompts, veto rule)
  audio/       trigger.py (EfficientAT), home_head.py + train_home_head.py, clap_scorer.py, pipeline.py (fusion)
  pipeline/    run_video.py (end-to-end CLI on one video)
  data/, eval/ dataset loaders, metrics, Stage 2 and audio evaluations
webui/         server.py (FastAPI + WebSocket), engine.py, static/ (front end)
scripts/       ucf_download.py, extract_x3d.py, xdv_download.py, fetch_models.py
colab/         the Colab jobs used for feature extraction and training (+ README runbook)
notebooks/     Colab notebook for download + extraction (fallback to the CLI jobs)
web_scraping/  scraping_file.py (YouTube + Instagram), clip_summarize.py (CLIP pre-labelling), scraped link lists
data/          splits/ (official UCF-Crime splits + temporal annotations), manifests/ (our 1,000-video subset),
               audio_eval/results/ (evaluation JSONs behind the reported numbers)
report/        final report (PDF + LaTeX)
demo_media/    how the demo clips were made (media files not committed)
```

## Setup

Not committed on purpose: model weights, trained checkpoints, datasets, extracted features, demo media and the
virtual environment. The steps below recreate everything.

### 1. Environment

```bash
git clone https://github.com/Nahushkashyap/data-mining-video-anamoly-detection.git
cd data-mining-video-anamoly-detection
python3 -m venv venv            # run_webui.sh expects the venv at ./venv
source venv/bin/activate
pip install -r requirements.txt
pip install msclap==1.3.4 --no-deps
```
`ffmpeg` must be on the PATH (`brew install ffmpeg` on macOS) for audio extraction from videos.

### 2. Pretrained models (frozen backbones, about 2 GB, one time)

```bash
python scripts/fetch_models.py          # or: python scripts/fetch_models.py x3d clip clap efficientat
```

| Model | Source | Stored in |
|---|---|---|
| X3D-S (Kinetics-400) | `torch.hub` `facebookresearch/pytorchvideo`, `x3d_s` | `models_cache/torch/` |
| CLIP ViT-B/16 | Hugging Face `openai/clip-vit-base-patch16` | `models_cache/huggingface/` |
| MS-CLAP 2023 + GPT-2 text encoder | Hugging Face `microsoft/msclap` (`CLAP_weights_2023.pth`), `gpt2` | `models_cache/huggingface/` |
| EfficientAT mn10 (AudioSet, mAP 0.471) | [fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT): code (`models/`, `helpers/`, `metadata/`) + release `v0.0.1/mn10_as_mAP_471.pt` | `models_cache/efficientat/` (code in `src/`) |

At run time the app is fully offline (`HF_HUB_OFFLINE=1` in `run_webui.sh`). All caches stay inside the project folder.

### 3. Trained checkpoints

| File | What | How to get it |
|---|---|---|
| `checkpoints/best_seed416.pt` | Video Stage 1 MIL head (τ = 0.350 stored inside) | **Required.** Train it (step 4) or ask the authors for the file. |
| `models_cache/efficientat/home_head_v1.pt` | Home-security audio head | Optional. Without it (or with `VAD_AUDIO_MODEL=zeroshot`) the app uses the zero-shot AudioSet trigger. Train it with `colab/job_audio_home.py`. |

### 4. Reproducing training (optional)

All heavy steps were run on Google Colab (T4) with the [Colab CLI](colab/README.md). Each command below is also
documented at the top of its script.

**UCF-Crime subset and X3D-S features.** The 1,000-video subset is defined by `data/manifests/ucf_subset.csv`
(lanes L1 + L2). Videos are fetched one at a time from the Hugging Face mirror `jinmang2/ucf_crime`. A Hugging Face token
is optional and only avoids rate limits: `export HF_TOKEN=...`.
```bash
python scripts/extract_x3d.py --lanes L1,L2 --with-download --out data/features/x3d_s   # or: colab/job_extract.py
```
**Stage 1 (3 seeds, picks τ on the hold-out split, evaluates on the 226 test videos):**
```bash
python -m vad.stage1.train --features data/features/x3d_s --out runs/stage1_sultani
cp runs/stage1_sultani/best_seed416.pt checkpoints/
```
**Stage 2 evaluation (CLIP on UCF-Crime test windows):**
```bash
python -m vad.eval.eval_stage2 --videos <ucf_videos_dir> --features data/features/x3d_s \
    --checkpoint checkpoints/best_seed416.pt --out runs/stage2
```
**XD-Violence audio-visual fusion:** `scripts/xdv_download.py`, then `python -m vad.stage1.train_xdv --xdv-root <dir> --out runs/xdv`
(the whole sequence is `colab/job_stage1_and_xdv.py`).
**Home-security audio head (FSD50K):** `python -m vad.audio.train_home_head {select,download,extract,train,clap} ...`.
The exact sequence and arguments are in `colab/job_audio_home.py`, and the runbook is in `colab/README.md`.

## Run

**Web app**
```bash
./run_webui.sh                  # http://127.0.0.1:8000
```
Upload an `.mp4/.mov/.avi` or switch to **Live camera**. Sliders: Stage 1 threshold τ, CLIP/CLAP veto threshold,
audio trigger τₐ. Each alert shows the CLIP/CLAP class probabilities and a per-frame anomaly curve, and **Inspect**
shows the raw embeddings and hidden activations.
Environment switches: `VAD_CHECKPOINT=<path>`, `VAD_DEVICE=cpu|mps|cuda`, `VAD_AUDIO_MODEL=zeroshot`, `PORT=8001`.

**Command line (one video)**
```bash
PYTHONPATH=. python -m vad.pipeline.run_video --video path/to/video.mp4 \
    --checkpoint checkpoints/best_seed416.pt --out-json result.json
```

## Web data mining

```bash
cd web_scraping
pip install -r requirements.txt
export APIFY_API_TOKEN=...      # only needed for the Instagram part
python scraping_file.py         # YouTube (yt-dlp search) + Instagram hashtags (Apify) -> scraped_*.json
python clip_summarize.py --limit 5   # CLIP pre-labelling: 16 frames per video, class + synthetic-footage score
```
The committed `scraped_raw_data.json` (367 YouTube) and `scraped_instagram_*.json` (122 reels) are the 489 links
described in the report. `clip_summaries.csv` holds the pilot pre-labelling. The mined videos were **not** used for
training, because they still need manual review.

## Notes and limitations
- UCF-Crime numbers are on our 226-video test subset (6 crime classes + normal), not the full 290-video test set,
  so they are not a like-for-like ranking against published results.
- Masked-person detection is untested (no labelled data). Alerts appear only in the app (no push notifications).
- See the report's Section VI for the full discussion.

## References
Sultani et al., CVPR 2018 (base paper) · Feichtenhofer, X3D, CVPR 2020 · Radford et al., CLIP, ICML 2021 ·
Schmid et al., EfficientAT, ICASSP 2023 · Elizalde et al., MS-CLAP, ICASSP 2024 · Fonseca et al., FSD50K, TASLP 2022 ·
Wu et al., XD-Violence, ECCV 2020. The full list is in the report.
