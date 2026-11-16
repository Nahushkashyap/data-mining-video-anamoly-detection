"""Colab foreground job: Stage 2 CLIP ViT-B/16 on 3 test videos + full Stage 2 evaluation.

Run from the project folder on the Mac:
    caffeinate -i venv/bin/colab exec -s vad --timeout 0 -f colab/job_stage2.py

Expects `/content/vad_bundle.tar.gz` uploaded and Google Drive mounted at `/content/drive`.
Results and logs go to `MyDrive/vad/runs/stage2_clip/`.
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

SAMPLE_VIDEOS = [
    ("Arson007_x264", "Test/Arson/Arson007_x264.mp4"),
    ("Explosion002_x264", "Test/Explosion/Explosion002_x264.mp4"),
    ("Robbery048_x264", "Test/Robbery/Robbery048_x264.mp4"),
]


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

    s1_dir = DRIVE_ROOT / "runs" / "stage1_sultani"
    s1_summary = json.loads((s1_dir / "summary.json").read_text(encoding="utf-8"))
    # Select checkpoint strictly by highest hold-out video AUC (zero test leakage)
    best_seed_rec = max(s1_summary["per_seed"], key=lambda r: float(r["holdout_video_auc"]))
    ckpt_path = s1_dir / f"best_seed{best_seed_rec['seed']}.pt"

    out_dir = DRIVE_ROOT / "runs" / "stage2_clip"
    samples_dir = out_dir / "samples"
    peaks_dir = samples_dir / "peaks"
    samples_dir.mkdir(parents=True, exist_ok=True)
    peaks_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stage2_log.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    videos_root = DRIVE_ROOT / "ucf_videos"
    feats_root = DRIVE_ROOT / "features" / "x3d_s"
    manifest = WORK_DIR / "data" / "manifests" / "ucf_subset.csv"
    annotations = WORK_DIR / "data" / "splits" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

    with open(log_path, "a", encoding="utf-8") as log:
        header = (
            f"\n================ {time.strftime('%Y-%m-%d %H:%M:%S')} Stage 2 CLIP job ================\n"
            f"Selected Stage 1 checkpoint by highest hold-out AUC: {ckpt_path.name} "
            f"(holdout_auc={best_seed_rec['holdout_video_auc']:.4f}, tau={best_seed_rec['tau_info']['tau']:.4f})\n"
        )
        print(header, end="", flush=True)
        log.write(header)
        log.flush()

        import torch
        print(
            f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            flush=True,
        )

        # 1. Run end-to-end run_video.py on 3 diverse test anomaly videos
        for vid, rel_path in SAMPLE_VIDEOS:
            run_logged(
                [
                    sys.executable, "-u", "-m", "vad.pipeline.run_video",
                    "--video", str(videos_root / rel_path),
                    "--checkpoint", str(ckpt_path),
                    "--features", str(feats_root / f"{vid}.npy"),
                    "--peaks-dir", str(peaks_dir),
                    "--out-json", str(samples_dir / f"{vid}.json"),
                    "--include-suppressed",
                    "--device", "auto",
                ],
                log,
                cwd=WORK_DIR,
                env=env,
            )

        # Print sample JSON summaries into the log
        for vid, _ in SAMPLE_VIDEOS:
            p = samples_dir / f"{vid}.json"
            if p.exists():
                txt = f"\n--- sample JSON {vid}.json ---\n" + p.read_text(encoding="utf-8") + "\n"
                print(txt, end="", flush=True)
                log.write(txt)
                log.flush()

        # 2. Full Stage 2 evaluation on GT anomaly windows + Stage 1 flagged windows (+ normal veto check)
        run_logged(
            [
                sys.executable, "-u", "-m", "vad.eval.eval_stage2",
                "--videos", str(videos_root),
                "--features", str(feats_root),
                "--checkpoint", str(ckpt_path),
                "--manifest", str(manifest),
                "--annotations", str(annotations),
                "--out", str(out_dir),
                "--device", "auto",
            ],
            log,
            cwd=WORK_DIR,
            env=env,
        )

        done_msg = f"\n================ STAGE 2 ALL DONE ({time.strftime('%H:%M:%S')}) ================\n"
        print(done_msg, end="", flush=True)
        log.write(done_msg)


main()
