"""Colab foreground job: UCF feature check -> Stage 1 (3 seeds) -> XD-Violence download & 3-head training.

Run from the project folder on the Mac:
    caffeinate -i venv/bin/colab exec -s vad --timeout 0 -f colab/job_stage1_and_xdv.py

Expects `/content/vad_bundle.tar.gz` uploaded via `colab upload` and Google Drive mounted at `/content/drive`.
All checkpoints, JSON summaries, archives, and logs are saved under `MyDrive/vad/`.
"""
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("VAD_DRIVE_ROOT", "/content/drive/MyDrive/vad"))
WORK_DIR = Path("/content/vad_work")
BUNDLE = Path("/content/vad_bundle.tar.gz")
LOCAL_XDV = Path("/content/xdv_local")


def run_logged(cmd: list[str], log_file, cwd: Path | None = None, env: dict | None = None) -> None:
    banner = f"\n>>> [{time.strftime('%H:%M:%S')}] {' '.join(cmd)}\n"
    print(banner, end="", flush=True)
    log_file.write(banner)
    log_file.flush()
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"Command failed (rc={rc}): {' '.join(cmd)}")


def main() -> None:
    if not DRIVE_ROOT.parent.is_dir():
        raise SystemExit(f"{DRIVE_ROOT.parent} not found: run `colab drivemount -s vad` first")
    if not BUNDLE.exists():
        raise SystemExit(f"{BUNDLE} missing: upload vad_bundle.tar.gz first")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as tf:
        tf.extractall(WORK_DIR)

    runs_dir = DRIVE_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "stage1_xdv_log.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "a", encoding="utf-8") as log:
        header = f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} Stage1 + XDV job ================\n"
        print(header, end="", flush=True)
        log.write(header)
        log.flush()

        import torch
        gpu_msg = f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
        print(gpu_msg, end="", flush=True)
        log.write(gpu_msg)

        # Install remotezip if missing (for XD-Violence I3D RGB center-crop range fetch)
        run_logged([sys.executable, "-m", "pip", "-q", "install", "remotezip"], log)

        # 1. Step 1 final check: verify all 1,000 UCF .npy files are [n, 2048] and finite
        ucf_feats = DRIVE_ROOT / "features" / "x3d_s"
        ucf_manifest = WORK_DIR / "data" / "manifests" / "ucf_subset.csv"
        ucf_ann = WORK_DIR / "data" / "splits" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
        run_logged(
            [
                sys.executable, "-u", "-m", "vad.data.ucf_dataset",
                "--features", str(ucf_feats),
                "--manifest", str(ucf_manifest),
                "--lanes", "L1,L2",
            ],
            log,
            cwd=WORK_DIR,
            env=env,
        )

        # 2. Step 3: Train UCF-Crime Stage 1 MIL head (3 seeds, 3,000 steps)
        s1_out = runs_dir / "stage1_sultani"
        run_logged(
            [
                sys.executable, "-u", "-m", "vad.stage1.train",
                "--features", str(ucf_feats),
                "--manifest", str(ucf_manifest),
                "--annotations", str(ucf_ann),
                "--lanes", "L1,L2",
                "--out", str(s1_out),
                "--seeds", "415,416,417",
                "--steps", "3000",
                "--batch-size", "30",
                "--lr", "1e-3",
                "--weight-decay", "1e-3",
                "--lambda-smooth", "8e-5",
                "--lambda-sparse", "8e-5",
                "--eval-every", "50",
                "--log-every", "500",
                "--device", "auto",
            ],
            log,
            cwd=WORK_DIR,
            env=env,
        )

        # 3. Step 5: Download XD-Violence VGGish (0.24 GB) + I3D RGB center-crop (4.25 GB) to fast local SSD
        xdv_drive = DRIVE_ROOT / "xdv"
        xdv_drive.mkdir(parents=True, exist_ok=True)
        xdv_tar = xdv_drive / "xdv_vggish_i3d_rgb0.tar"

        if xdv_tar.exists() and not (LOCAL_XDV / "annotations.txt").exists():
            msg = f"\n>>> Restoring cached XD-Violence archive from {xdv_tar} -> {LOCAL_XDV}\n"
            print(msg, end="", flush=True)
            log.write(msg)
            LOCAL_XDV.mkdir(parents=True, exist_ok=True)
            with tarfile.open(xdv_tar, "r:") as tf:
                tf.extractall(LOCAL_XDV)

        run_logged(
            [
                sys.executable, "-u", str(WORK_DIR / "scripts" / "xdv_download.py"),
                "--out", str(LOCAL_XDV),
                "--workers", "16",
            ],
            log,
            cwd=WORK_DIR,
            env=env,
        )

        # Persist XD-Violence features to Drive as one fast sequential tar (avoids 9,508 FUSE RPCs)
        if not xdv_tar.exists():
            t0 = time.time()
            msg = f"\n>>> Archiving {LOCAL_XDV} -> {xdv_tar} (single sequential write to Drive)...\n"
            print(msg, end="", flush=True)
            log.write(msg)
            tmp_tar = xdv_drive / "xdv_vggish_i3d_rgb0.tar.tmp"
            with tarfile.open(tmp_tar, "w:") as tf:
                for sub in ("vggish", "i3d_rgb", "annotations.txt", "annotations_aligned.txt", "gt.npy", "list"):
                    p = LOCAL_XDV / sub
                    if p.exists():
                        tf.add(p, arcname=sub)
            os.replace(tmp_tar, xdv_tar)
            gb = xdv_tar.stat().st_size / 1e9
            done_msg = f">>> Archived {gb:.2f} GB to {xdv_tar} in {time.time() - t0:.1f}s\n"
            print(done_msg, end="", flush=True)
            log.write(done_msg)

        # 4. Step 5 training: Video-only (I3D), Audio-only (VGGish), Late-fusion (score avg), 3 seeds
        xdv_out = runs_dir / "xdv_audio_visual"
        run_logged(
            [
                sys.executable, "-u", "-m", "vad.stage1.train_xdv",
                "--xdv-root", str(LOCAL_XDV),
                "--out", str(xdv_out),
                "--seeds", "415,416,417",
                "--steps", "3000",
                "--batch-size", "30",
                "--lr", "1e-3",
                "--weight-decay", "1e-3",
                "--lambda-smooth", "8e-5",
                "--lambda-sparse", "8e-5",
                "--eval-every", "100",
                "--log-every", "500",
                "--device", "auto",
            ],
            log,
            cwd=WORK_DIR,
            env=env,
        )

        done_all = f"\n================ ALL DONE ({time.strftime('%H:%M:%S')}) ================\n"
        print(done_all, end="", flush=True)
        log.write(done_all)


main()
