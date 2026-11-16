"""Colab foreground job: Real Life Violence Situations part of the audio-datasets job only (rerun 20 Nov).

Run after colab/job_audio_datasets.py in the same session (reuses /content/vad_work and installed packages):
    caffeinate -i venv/bin/colab exec -s vad --timeout 0 -f colab/job_rlvs.py

On Colab, kagglehub serves RLVS from the built-in /kaggle/input mount rather than ~/.cache/kagglehub.
Any failure is written to the Drive log as 'Command failed' so colab/watchdog_stop.sh sees it.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/vad")
WORK = Path("/content/vad_work")
OUT = DRIVE_ROOT / "runs" / "audio_datasets"
S1 = DRIVE_ROOT / "runs" / "stage1_sultani" / "best_seed416.pt"


def find_rlvs() -> Path:
    import kagglehub
    base = Path(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"))
    # the archive holds the dataset twice; use the first 'Real Life Violence Dataset' dir with Violence/NonViolence
    for d in sorted(base.rglob("Real Life Violence Dataset"), key=lambda p: len(p.parts)):
        if (d / "Violence").is_dir() and (d / "NonViolence").is_dir():
            return d
    raise FileNotFoundError(f"no Violence/NonViolence dirs under {base}")


def main():
    log = open(OUT / "audio_datasets_log.txt", "a", encoding="utf-8")
    log.write(f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} RLVS rerun ================\n")
    log.flush()
    try:
        root = find_rlvs()
        log.write(f"[rlvs] root {root}\n")
        log.flush()
        env = os.environ.copy()
        env.update(PYTHONPATH=str(WORK), PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", HF_HUB_DISABLE_PROGRESS_BARS="1")
        cmd = [sys.executable, "-u", "-m", "vad.eval.eval_audio_datasets", "rlvs", "--root", str(root), "--out", str(OUT),
               "--checkpoint", str(S1), "--feat-dir", "/content/rlvs_feats"]
        p = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            if any(s in line for s in ("Loading weights", "Warning: FMAX")):
                continue
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if p.wait() != 0:
            raise RuntimeError(f"rlvs eval rc={p.returncode}")
    except BaseException as exc:
        log.write(f"Command failed: {type(exc).__name__}: {exc}\n")
        log.close()
        raise
    log.write(f"\n================ AUDIO DATASETS JOB DONE ({time.strftime('%H:%M:%S')}) ================\n")
    log.close()
    print("RLVS done", flush=True)


main()
