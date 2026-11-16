"""UCF-Crime download + X3D-S extraction on a Colab VM, driven by the Colab CLI.

Run from the project folder on the Mac (see colab/README.md for the full sequence):
    caffeinate -i colab exec -s vad --timeout 0 -f colab/job_extract.py

`colab exec -f` sends this file's code to the VM's Jupyter kernel. The job runs
in the foreground on purpose: the Colab backend keeps a VM alive only while its
kernel is busy, so a background job could be reclaimed mid-run.

Everything is written to Google Drive (videos, features, logs) and every step is
resumable, so if the VM or the connection dies, just run the same command again
(on a new session if needed) and it carries on from the last saved clip.
"""
import os
import subprocess
import sys
import time

# ---- settings -------------------------------------------------------------
LANES = "L1,L2"           # L1 = test set + train anomalies (600), L2 = 400 train normals; "L2X" = 400 more normals
DOWNLOAD_WORKERS = 12     # parallel range-fetch downloads
DELETE_AFTER = False      # keep raw videos on Drive (their audio tracks may be needed later)
DRIVE_ROOT = os.environ.get("VAD_DRIVE_ROOT", "/content/drive/MyDrive/vad")
SCRIPTS = os.environ.get("VAD_SCRIPTS", "/content")  # where `colab upload` put the two scripts
# ---------------------------------------------------------------------------


def _check():
    if not os.path.isdir(os.path.dirname(DRIVE_ROOT)):
        raise SystemExit(f"{os.path.dirname(DRIVE_ROOT)} not found: run `colab drivemount -s vad` first")
    for f in ("ucf_download.py", "extract_x3d.py"):
        if not os.path.exists(os.path.join(SCRIPTS, f)):
            raise SystemExit(f"{SCRIPTS}/{f} missing: run the two `colab upload` commands first")
    os.makedirs(DRIVE_ROOT, exist_ok=True)


def _setup():
    if os.environ.get("VAD_SKIP_PIP"):
        return
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "remotezip", "fvcore"], check=True)
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "none (extracting on CPU; slower, but resumable on a GPU session later)", flush=True)


def _run():
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS, "extract_x3d.py"),
           "--lanes", LANES, "--with-download", "--no-disk-check",
           "--download-workers", str(DOWNLOAD_WORKERS), "--worker", "colab-cli", "--device", "auto",
           "--videos", f"{DRIVE_ROOT}/ucf_videos", "--splits", f"{DRIVE_ROOT}/splits",
           "--manifest", f"{DRIVE_ROOT}/manifests/ucf_subset.csv", "--out", f"{DRIVE_ROOT}/features/x3d_s"]
    if DELETE_AFTER:
        cmd.append("--delete-after")
    log_path = os.path.join(DRIVE_ROOT, "run_log.txt")
    with open(log_path, "a") as log:
        stamp = f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} lanes={LANES} =====\n"
        log.write(stamp)
        print(stamp.strip(), f"(also logged to {log_path})", flush=True)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = p.wait()
        log.write(f"===== extractor exit code {rc} =====\n")
    print(f"extractor exit code {rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


_check()
_setup()
_run()
