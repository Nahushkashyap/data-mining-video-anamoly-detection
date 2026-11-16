"""Colab foreground job: train the home-security audio head (scream / glass / loud bang / gunshot / explosion / alarm).

Run from the project folder on the Mac (session with a T4; no Drive needed):
    venv/bin/colab upload -s vadaudio /tmp/vad_home_bundle.tar.gz /content/vad_home_bundle.tar.gz
    venv/bin/colab upload -s vadaudio home_cfg.json /content/home_cfg.json      # {"tag": "full", "limit": null, "epochs": 40}
    caffeinate -i venv/bin/colab exec -s vadaudio --timeout 0 -f colab/job_audio_home.py
    venv/bin/colab download -s vadaudio /content/home_out/full/home_head_v1.pt models_cache/efficientat/home_head_v1.pt

Bundle = vad/, models_cache/efficientat/, scripts/ucf_download.py (its HF_TOKEN, or the HF_TOKEN env var, authenticates the clip downloads on
this VM; it is written to a 0600 file and never printed). Downloads (approved 22 Nov): ~12k FSD50K clips from the
Hugging Face mirror Fhrozen/FSD50k to the VM's local disk only, plus MS-CLAP weights for the two-stage evaluation.
Outputs -> /content/home_out/<tag>/ (head weights, eval JSONs, log).
"""
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/content/vad_work")
BUNDLE = Path("/content/vad_home_bundle.tar.gz")
FSD = Path("/content/fsd")
CFG = json.loads(Path("/content/home_cfg.json").read_text()) if Path("/content/home_cfg.json").exists() else {}
TAG = CFG.get("tag", "full")
OUT = Path("/content/home_out") / TAG
LABELS_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/labels/"


def run(cmd, log, cwd=None, env=None, check=True):
    banner = f"\n>>> [{time.strftime('%H:%M:%S')}] {' '.join(map(str, cmd))}\n"
    print(banner, end="", flush=True)
    log.write(banner)
    log.flush()
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        if any(s in line for s in ("Loading weights", "Warning: FMAX", "warnings.warn", "UserWarning", "FutureWarning")):
            continue
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    rc = p.wait()
    if rc != 0 and check:
        raise SystemExit(f"Command failed (rc={rc})")
    return rc


def main():
    if not BUNDLE.exists():
        raise SystemExit("bundle missing: upload /content/vad_home_bundle.tar.gz first")
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as tf:
        tf.extractall(WORK, filter="data")
    OUT.mkdir(parents=True, exist_ok=True)
    (FSD / "labels").mkdir(parents=True, exist_ok=True)
    tok_file = Path("/content/.hf_token")
    m = re.search(r'^HF_TOKEN\s*=\s*["\']([^"\']+)["\']', (WORK / "scripts" / "ucf_download.py").read_text(), re.M)
    token = m.group(1) if m else os.environ.get("HF_TOKEN", "")
    if token:
        tok_file.write_text(token)
        tok_file.chmod(0o600)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(WORK), PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", HF_HUB_DISABLE_PROGRESS_BARS="1")
    py = [sys.executable, "-u", "-m", "vad.audio.train_home_head"]
    manifest, npz = OUT / "manifest.json", OUT / "features.npz"
    with open(OUT / "log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} home audio job ({CFG}) ================\n")
        run([sys.executable, "-m", "pip", "install", "-q", "torchlibrosa"], log)
        run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "msclap"], log)
        for f in ("dev.csv", "eval.csv", "vocabulary.csv"):
            if not (FSD / "labels" / f).exists():
                run(["curl", "-sfL", "--retry", "5", "-o", FSD / "labels" / f, LABELS_URL + f], log)
        sel = py + ["select", "--labels", FSD / "labels", "--manifest", manifest]
        if CFG.get("limit"):
            sel += ["--limit", CFG["limit"]]
        run(sel, log, cwd=WORK, env=env)
        run(py + ["download", "--manifest", manifest, "--root", FSD / "clips", "--token-file", tok_file], log, cwd=WORK, env=env)
        n_wav = sum(1 for _ in (FSD / "clips").rglob("*.wav"))
        du = subprocess.run(["du", "-sh", str(FSD / "clips")], capture_output=True, text=True).stdout.strip()
        msg = f"[job] clips on disk: {n_wav} ({du})\n"
        print(msg, end="", flush=True)
        log.write(msg)
        run(py + ["extract", "--manifest", manifest, "--root", FSD / "clips", "--npz", npz], log, cwd=WORK, env=env)
        run(py + ["train", "--npz", npz, "--out", OUT, "--epochs", CFG.get("epochs", 40)], log, cwd=WORK, env=env)
        if CFG.get("clap", True):
            rc = run(py + ["clap", "--out", OUT, "--root", FSD / "clips"], log, cwd=WORK, env=env, check=False)
            if rc != 0:
                log.write("[job] two-stage CLAP evaluation failed (head results are still saved)\n")
        print(f"\n================ HOME AUDIO JOB DONE ({time.strftime('%H:%M:%S')}) -> {OUT} ================", flush=True)
        log.write("JOB DONE\n")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  {p.stat().st_size / 1e6:.2f} MB")


main()
