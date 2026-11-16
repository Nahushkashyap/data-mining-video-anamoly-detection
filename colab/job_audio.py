"""Colab foreground job: two-stage audio branch + audio/video fusion evaluation on the 226 UCF test videos.

Run from the project folder on the Mac:
    caffeinate -i venv/bin/colab exec -s vad --timeout 0 -f colab/job_audio.py

Expects `/content/vad_bundle.tar.gz` (vad/, data manifest + annotations, models_cache/efficientat/) and
Google Drive mounted at /content/drive. CLIP ViT-B/16, MS-CLAP 2023 and GPT-2 are fetched from Hugging
Face on the VM. Results/log go to MyDrive/vad/runs/audio_branch/.
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("VAD_DRIVE_ROOT", "/content/drive/MyDrive/vad"))
WORK_DIR = Path("/content/vad_work")
BUNDLE = Path("/content/vad_bundle.tar.gz")


def run_logged(cmd, log, cwd=None, env=None):
    banner = f"\n>>> [{time.strftime('%H:%M:%S')}] {' '.join(cmd)}\n"
    print(banner, end="", flush=True)
    log.write(banner)
    log.flush()
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "Loading weights" in line or "Redirects are currently" in line:
            continue
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if p.wait() != 0:
        raise SystemExit(f"Command failed (rc={p.returncode}): {' '.join(cmd)}")


def main():
    if not DRIVE_ROOT.parent.is_dir():
        raise SystemExit(f"{DRIVE_ROOT.parent} not found: run `colab drivemount -s vad` first")
    if not BUNDLE.exists():
        raise SystemExit(f"{BUNDLE} missing: upload vad_bundle.tar.gz first")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as tf:
        tf.extractall(WORK_DIR, filter="data")

    out_dir = DRIVE_ROOT / "runs" / "audio_branch"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1 = DRIVE_ROOT / "runs" / "stage1_sultani"
    summ = json.loads((s1 / "summary.json").read_text())
    best = max(summ["per_seed"], key=lambda r: float(r["holdout_video_auc"]))
    ckpt = s1 / f"best_seed{best['seed']}.pt"

    env = os.environ.copy()
    env.update(PYTHONPATH=str(WORK_DIR), PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", HF_HUB_DISABLE_PROGRESS_BARS="1")
    with open(out_dir / "audio_log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} audio branch job ({ckpt.name}) ================\n")
        run_logged([sys.executable, "-m", "pip", "install", "-q", "torchlibrosa"], log)
        run_logged([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "msclap"], log)
        run_logged([sys.executable, "-c", "import torch, torchaudio, librosa, transformers, msclap; "
                    "print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'librosa', librosa.__version__, "
                    "'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"],
                   log, env=env)
        run_logged([sys.executable, "-u", "-m", "vad.eval.eval_audio",
                    "--videos", str(DRIVE_ROOT / "ucf_videos"),
                    "--features", str(DRIVE_ROOT / "features" / "x3d_s"),
                    "--checkpoint", str(ckpt),
                    "--manifest", str(WORK_DIR / "data" / "manifests" / "ucf_subset.csv"),
                    "--annotations", str(WORK_DIR / "data" / "splits" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
                    "--out", str(out_dir), "--device", "auto"], log, cwd=WORK_DIR, env=env)
        msg = f"\n================ AUDIO JOB DONE ({time.strftime('%H:%M:%S')}) ================\n"
        print(msg, end="", flush=True)
        log.write(msg)


main()
