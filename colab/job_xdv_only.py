"""Colab foreground job: XD-Violence VGGish + I3D RGB center-crop download & 3-head training.

Run from the project folder on the Mac:
    caffeinate -i venv/bin/colab exec -s vad --timeout 0 -f colab/job_xdv_only.py
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
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as tf:
        tf.extractall(WORK_DIR)

    runs_dir = DRIVE_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "xdv_log.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "a", encoding="utf-8") as log:
        header = f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} XD-Violence Audio+Video job ================\n"
        print(header, end="", flush=True)
        log.write(header)
        log.flush()

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

        # Verify both VGGish and I3D counts before archiving
        n_v_tr = len(list((LOCAL_XDV / "i3d_rgb" / "train").glob("*.npy")))
        n_v_te = len(list((LOCAL_XDV / "i3d_rgb" / "test").glob("*.npy")))
        n_a_tr = len(list((LOCAL_XDV / "vggish" / "train").glob("*.npy")))
        n_a_te = len(list((LOCAL_XDV / "vggish" / "test").glob("*.npy")))
        check_msg = f"\n>>> Feature counts: I3D=({n_v_tr} tr, {n_v_te} te) | VGGish=({n_a_tr} tr, {n_a_te} te)\n"
        print(check_msg, end="", flush=True)
        log.write(check_msg)
        if n_v_tr < 3900 or n_v_te < 800 or n_a_tr < 3900 or n_a_te < 800:
            raise SystemExit(f"Incomplete XD-Violence features: {check_msg.strip()}")

        # Persist XD-Violence features to Drive as one fast sequential tar (recreate if smaller than 5 GB)
        if not xdv_tar.exists() or xdv_tar.stat().st_size < 5_000_000_000:
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
        done_all = f"\n================ XD-VIOLENCE ALL DONE ({time.strftime('%H:%M:%S')}) ================\n"
        print(done_all, end="", flush=True)
        log.write(done_all)


main()
