"""Colab foreground job: audio branch on TUT Rare Sound Events 2017 (eval mixtures) and Real Life Violence Situations.

Run from the project folder on the Mac (after colab/job_audio.py, same session):
    caffeinate -i venv/bin/colab exec -s vad --timeout 0 -f colab/job_audio_datasets.py

Expects /content/vad_bundle2.tar.gz (vad/, scripts/extract_x3d.py, models_cache/efficientat/) and Drive mounted.
Downloads (approved 20 Nov): TUT eval mixture zips (~4.6 GB, Zenodo record 1160455) and RLVS (~1.95 GB, Kaggle,
anonymous via kagglehub) to the VM's local disk only. Results/log -> MyDrive/vad/runs/audio_datasets/.
"""
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/vad")
WORK = Path("/content/vad_work")
BUNDLE = Path("/content/vad_bundle2.tar.gz")
TUT_DIR = Path("/content/tut")
ZENODO = "https://zenodo.org/records/1160455/files/"
TUT_FILES = ["TUT-rare-sound-events-2017-evaluation.doc.zip"] + [
    f"TUT-rare-sound-events-2017-evaluation.mixture_data.{i}.zip" for i in range(1, 6)]


def run(cmd, log, cwd=None, env=None):
    banner = f"\n>>> [{time.strftime('%H:%M:%S')}] {' '.join(map(str, cmd))}\n"
    print(banner, end="", flush=True)
    log.write(banner)
    log.flush()
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        if any(s in line for s in ("Loading weights", "Redirects are currently", "Warning: FMAX")):
            continue
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if p.wait() != 0:
        raise SystemExit(f"Command failed (rc={p.returncode})")


def main():
    if not DRIVE_ROOT.is_dir() or not BUNDLE.exists():
        raise SystemExit("Drive not mounted or bundle missing")
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as tf:
        tf.extractall(WORK, filter="data")
    out = DRIVE_ROOT / "runs" / "audio_datasets"
    out.mkdir(parents=True, exist_ok=True)
    s1 = DRIVE_ROOT / "runs" / "stage1_sultani" / "best_seed416.pt"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(WORK), PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", HF_HUB_DISABLE_PROGRESS_BARS="1")
    with open(out / "audio_datasets_log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} audio datasets job ================\n")
        run([sys.executable, "-m", "pip", "install", "-q", "torchlibrosa", "fvcore", "kagglehub"], log)
        run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "msclap"], log)

        # ---------------- TUT Rare Sound Events 2017 (evaluation mixtures) ----------------
        TUT_DIR.mkdir(exist_ok=True)
        for f in TUT_FILES:
            if not (TUT_DIR / f).exists():
                run(["curl", "-sfL", "--retry", "5", "-o", TUT_DIR / f, ZENODO + f + "?download=1"], log)
        for f in TUT_FILES:
            with zipfile.ZipFile(TUT_DIR / f) as z:
                z.extractall(TUT_DIR)
        n_wav = sum(1 for _ in TUT_DIR.rglob("*.wav"))
        n_ann = [p.name for p in TUT_DIR.rglob("event_list*")]
        msg = f"[tut] extracted {n_wav} wav files; annotation lists: {n_ann}\n"
        print(msg, end="", flush=True)
        log.write(msg)
        run([sys.executable, "-u", "-m", "vad.eval.eval_audio_datasets", "tut", "--root", TUT_DIR, "--out", out],
            log, cwd=WORK, env=env)

        # ---------------- Real Life Violence Situations (Kaggle) ----------------
        run([sys.executable, "-c", "import kagglehub; print('RLVS at', kagglehub.dataset_download("
             "'mohamedmustafa/real-life-violence-situations-dataset'))"], log, env=env)
        # on Colab kagglehub serves RLVS from /kaggle/input, not ~/.cache (failed 20 Nov); see colab/job_rlvs.py
        import kagglehub
        base = Path(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"))
        rlvs = next(d for d in sorted(base.rglob("Real Life Violence Dataset"), key=lambda p: len(p.parts))
                    if (d / "Violence").is_dir())
        run([sys.executable, "-u", "-m", "vad.eval.eval_audio_datasets", "rlvs", "--root", rlvs, "--out", out,
             "--checkpoint", s1, "--feat-dir", "/content/rlvs_feats"], log, cwd=WORK, env=env)
        done = f"\n================ AUDIO DATASETS JOB DONE ({time.strftime('%H:%M:%S')}) ================\n"
        print(done, end="", flush=True)
        log.write(done)


main()
