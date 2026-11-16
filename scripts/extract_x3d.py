#!/usr/bin/env python3
"""Extract X3D-S clip features from UCF-Crime videos *while they download*.

It watches the download folder and processes each video as soon as the
downloader finishes it (the downloader writes `<name>.part` and renames it when
complete, so a video with its final name and the manifest size is safe to read).
With --with-download it also starts scripts/ucf_download.py for you, so one
command does both and the GPU/MPS never waits for the whole download.

Per video it writes:
  <out>/<video_id>.npy   float16 [n_clips, 2048]  X3D-S penultimate features
  <out>/<video_id>.json  fps, n_frames, n_clips, clip/stride settings, timing

Clip definition (fixed, identical on Mac and Colab): every 6th frame is kept,
13 kept frames form one clip, so clip i covers raw frames [78*i, 78*i + 78)
(~2.6 s at 30 fps), non-overlapping; the last clip is padded with its final frame.
Frames are resized (short side 182), centre-cropped to 182x182 and normalised
with mean 0.45 / std 0.225 (PyTorchVideo X3D defaults).

Safe to stop and re-run: finished videos are skipped. A video in progress is
checkpointed every --ckpt-secs to <out>/_partial/<video_id>.npy, so a rerun (even
on a different machine or runtime, e.g. Colab CPU -> T4) continues from the last
saved clip instead of restarting the video. Claim files let several workers
share one feature folder without doing the same video twice.

Mac (download + process, one command):
  caffeinate -i venv/bin/python scripts/extract_x3d.py --lanes L1 --with-download
Process only (videos already downloaded, or downloader running elsewhere):
  venv/bin/python scripts/extract_x3d.py --lanes L1 --wait

First run needs:  pip install fvcore   (X3D is loaded via torch.hub from PyTorchVideo)
"""
import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

STRIDE = 6          # keep every 6th raw frame
CLIP_LEN = 13       # kept frames per clip  -> 78 raw frames per clip
SIZE = 182          # X3D-S crop size
MEAN, STD = 0.45, 0.225
CLAIM_TTL = 20 * 60  # seconds before another worker may take over a claimed video


# ---------------------------------------------------------------- model
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_x3d(device):
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True, trust_repo=True)
    head = model.blocks[-1]
    head.proj = torch.nn.Identity()          # drop the 400-way Kinetics classifier
    if getattr(head, "activation", None) is not None:
        head.activation = torch.nn.Identity()
    model = model.eval().to(device)
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = model.to(dtype)
    with torch.inference_mode():
        d = model(torch.zeros(1, 3, CLIP_LEN, SIZE, SIZE, device=device, dtype=dtype)).shape[-1]
    if d != 2048:
        raise RuntimeError(f"expected 2048-d features, got {d}")
    return model, dtype


# ---------------------------------------------------------------- video
def _prep(frame):
    h, w = frame.shape[:2]
    s = SIZE / min(h, w)
    frame = cv2.resize(frame, (max(SIZE, round(w * s)), max(SIZE, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    y, x = (h - SIZE) // 2, (w - SIZE) // 2
    return cv2.cvtColor(frame[y:y + SIZE, x:x + SIZE], cv2.COLOR_BGR2RGB)


def iter_clip_batches(path, batch, skip_clips=0):
    """Yields uint8 arrays [b, 13, 182, 182, 3]; returns (n_raw_frames, fps) via StopIteration value.

    skip_clips: the first N clips are already embedded (resume); their frames are
    only grabbed (cheap), not decoded to images or yielded.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_raw, clip, clips = 0, [], []
    skip_frames = skip_clips * CLIP_LEN * STRIDE
    while True:
        if not cap.grab():
            break
        if n_raw < skip_frames:
            n_raw += 1
            continue
        if n_raw % STRIDE == 0:
            ok, frame = cap.retrieve()
            if ok:
                clip.append(_prep(frame))
            if len(clip) == CLIP_LEN:
                clips.append(np.stack(clip))
                clip = []
                if len(clips) == batch:
                    yield np.stack(clips)
                    clips = []
        n_raw += 1
    cap.release()
    if clip:  # pad the final partial clip with its last frame
        clip += [clip[-1]] * (CLIP_LEN - len(clip))
        clips.append(np.stack(clip))
    if clips:
        yield np.stack(clips)
    return n_raw, fps


def load_partial(partial_path):
    """Features saved by an interrupted run, or None."""
    if not os.path.exists(partial_path):
        return None
    try:
        arr = np.load(partial_path)
        if arr.ndim == 2 and arr.shape[1] == 2048 and len(arr) > 0 and np.isfinite(arr).all():
            return arr.astype(np.float32)
    except Exception:
        pass
    os.remove(partial_path)  # unreadable / wrong shape: start the video over
    return None


def save_partial(partial_path, feats):
    tmp = partial_path[:-4] + ".tmp.npy"
    np.save(tmp, np.concatenate(feats).astype(np.float16))
    os.replace(tmp, partial_path)


def embed_video(model, dtype, device, path, batch, partial_path=None, ckpt_secs=120):
    """Returns (feats fp16 [n_clips, 2048], n_raw_frames, fps, resumed_clips)."""
    prev = load_partial(partial_path) if partial_path else None
    feats = [prev] if prev is not None else []
    skip = len(prev) if prev is not None else 0
    gen = iter_clip_batches(path, batch, skip_clips=skip)
    last_ckpt = time.time()
    mean = torch.tensor(MEAN, device=device, dtype=dtype)
    std = torch.tensor(STD, device=device, dtype=dtype)
    while True:
        try:
            arr = next(gen)
        except StopIteration as stop:
            n_raw, fps = stop.value
            break
        x = torch.from_numpy(arr).to(device)                       # b,t,h,w,c uint8
        x = x.permute(0, 4, 1, 2, 3).to(dtype).div_(255).sub_(mean).div_(std)  # b,c,t,h,w
        with torch.inference_mode():
            feats.append(model(x).float().cpu().numpy())
        if partial_path and time.time() - last_ckpt >= ckpt_secs:
            save_partial(partial_path, feats)
            last_ckpt = time.time()
    if not feats:
        raise IOError("no frames decoded")
    out = np.concatenate(feats).astype(np.float16)
    expected = -(-n_raw // (CLIP_LEN * STRIDE))  # ceil
    if skip and len(out) != expected:
        raise IOError(f"clip count {len(out)} != expected {expected} (stale partial?)")
    return out, n_raw, fps, skip


# ---------------------------------------------------------------- bookkeeping
def load_rows(manifest, lanes):
    with open(manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["lane"] in lanes]
    order = {"test": 0, "train": 1}
    rows.sort(key=lambda r: (order[r["split"]], -int(r["label"]), r["video_id"]))  # same order as the downloader
    return rows


def video_ready(path, size):
    return os.path.exists(path) and os.path.getsize(path) == size


def try_claim(claim_dir, vid, worker):
    p = os.path.join(claim_dir, vid + ".claim")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < CLAIM_TTL:
        try:
            if open(p).read().split()[0] != worker:
                return False
        except Exception:
            return False
    with open(p, "w") as f:
        f.write(f"{worker} {time.time():.0f}\n")
    return True


def save_atomic(out_dir, vid, feats, meta):
    tmp = os.path.join(out_dir, vid + ".tmp.npy")
    np.save(tmp, feats)
    os.replace(tmp, os.path.join(out_dir, vid + ".npy"))
    with open(os.path.join(out_dir, vid + ".json.tmp"), "w") as f:
        json.dump(meta, f)
    os.replace(os.path.join(out_dir, vid + ".json.tmp"), os.path.join(out_dir, vid + ".json"))


# ---------------------------------------------------------------- main loop
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", default="L1", help="comma list of L1,L2,L2X")
    ap.add_argument("--videos", default=os.path.join(proj, "ucf_videos.nosync"), help="download root")
    ap.add_argument("--manifest", default=os.path.join(proj, "data", "manifests", "ucf_subset.csv"))
    ap.add_argument("--splits", default=os.path.join(proj, "data", "splits"))
    ap.add_argument("--out", default=os.path.join(proj, "data", "features", "x3d_s"))
    ap.add_argument("--device", default="auto", help="auto / mps / cuda / cpu")
    ap.add_argument("--batch", type=int, default=0, help="clips per forward pass (default 16 mps, 32 cuda, 8 cpu)")
    ap.add_argument("--worker", default=socket.gethostname().split(".")[0])
    ap.add_argument("--with-download", action="store_true", help="also start ucf_download.py for the same lanes")
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--no-disk-check", action="store_true", help="passed to the downloader (Drive mounts)")
    ap.add_argument("--wait", action="store_true", help="keep polling for new videos until the lanes are complete")
    ap.add_argument("--poll", type=int, default=20, help="seconds between checks while waiting")
    ap.add_argument("--idle-timeout", type=int, default=45 * 60,
                    help="with --wait and no --with-download: stop after this many seconds without new videos")
    ap.add_argument("--delete-after", action="store_true", help="delete each video once its features are saved")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (testing)")
    ap.add_argument("--ckpt-secs", type=int, default=120,
                    help="save a video's partial features this often, so an interrupted video resumes mid-way")
    a = ap.parse_args()
    lanes = set(a.lanes.split(","))

    dl = None
    if a.with_download:
        cmd = [sys.executable, "-u", os.path.join(here, "ucf_download.py"), "--lanes", a.lanes, "--out", a.videos,
               "--splits", a.splits, "--manifest", a.manifest, "--workers", str(a.download_workers)]
        if a.no_disk_check:
            cmd.append("--no-disk-check")
        os.makedirs(a.videos, exist_ok=True)
        dl_log = open(os.path.join(a.videos, "downloader_stdout.txt"), "a")
        dl = subprocess.Popen(cmd, stdout=dl_log, stderr=subprocess.STDOUT)
        print(f"downloader started (pid {dl.pid}), log: {dl_log.name}", flush=True)
        a.wait = True

    def downloading():
        """Our downloader child is alive, or another one (e.g. from an earlier,
        interrupted run) holds a fresh lock on the video folder."""
        if dl is not None and dl.poll() is None:
            return True
        try:
            return time.time() - os.path.getmtime(os.path.join(a.videos, ".downloader.lock")) < 120
        except OSError:
            return False

    # the downloader builds the manifest on first run; wait for it
    while not os.path.exists(a.manifest):
        if dl is not None and not downloading():
            sys.exit("downloader exited before writing the manifest; see downloader_stdout.txt")
        if dl is None:
            sys.exit(f"manifest not found: {a.manifest} (run ucf_download.py once, or use --with-download)")
        time.sleep(3)

    rows = load_rows(a.manifest, lanes)
    os.makedirs(a.out, exist_ok=True)
    claim_dir = os.path.join(a.out, "_claims")
    os.makedirs(claim_dir, exist_ok=True)
    partial_dir = os.path.join(a.out, "_partial")
    os.makedirs(partial_dir, exist_ok=True)
    log = open(os.path.join(a.out, f"_log_{a.worker}.jsonl"), "a")

    device = pick_device(a.device)
    batch = a.batch or {"cuda": 32, "mps": 16}.get(device.type, 8)
    print(f"loading X3D-S on {device} (batch {batch})...", flush=True)
    model, dtype = load_x3d(device)

    n_done_run, t_start, clips_run, model_secs = 0, time.time(), 0, 0.0
    last_progress = time.time()
    failures = {}
    while True:
        done = pending = 0
        progressed = False
        for r in rows:
            vid = r["video_id"]
            if os.path.exists(os.path.join(a.out, vid + ".npy")):
                done += 1
                continue
            path = os.path.join(a.videos, r["rel_path"])
            if not video_ready(path, int(r["bytes"])) or failures.get(vid, 0) >= 2:
                pending += 1
                continue
            if not try_claim(claim_dir, vid, a.worker):
                pending += 1
                continue
            t0 = time.time()
            partial_path = os.path.join(partial_dir, vid + ".npy")
            try:
                feats, n_raw, fps, resumed = embed_video(model, dtype, device, path, batch,
                                                         partial_path, a.ckpt_secs)
            except Exception as e:
                failures[vid] = failures.get(vid, 0) + 1
                if os.path.exists(partial_path):
                    os.remove(partial_path)  # retry this video from scratch
                print(f"  FAIL {vid}: {type(e).__name__}: {e}", flush=True)
                log.write(json.dumps({"video_id": vid, "status": "error", "error": str(e)}) + "\n")
                log.flush()
                pending += 1
                continue
            dt = time.time() - t0
            meta = {"video_id": vid, "split": r["split"], "label": int(r["label"]), "cls": r["cls"],
                    "fps": fps, "n_frames": n_raw, "n_clips": int(feats.shape[0]), "stride": STRIDE,
                    "clip_len": CLIP_LEN, "size": SIZE, "worker": a.worker, "device": device.type,
                    "seconds": round(dt, 2), "resumed_clips": resumed}
            save_atomic(a.out, vid, feats, meta)
            for p in (os.path.join(claim_dir, vid + ".claim"), partial_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            if a.delete_after:
                os.remove(path)
            log.write(json.dumps({**meta, "status": "ok"}) + "\n")
            log.flush()
            done += 1
            n_done_run += 1
            new_clips = feats.shape[0] - resumed
            clips_run += new_clips
            model_secs += dt
            progressed = True
            note = f"  (resumed at clip {resumed})" if resumed else ""
            print(f"[{done}/{len(rows)}] {vid:28s} {feats.shape[0]:5d} clips  {dt:6.1f}s  "
                  f"{1000 * dt / max(new_clips, 1):5.1f} ms/clip{note}", flush=True)
            if a.limit and n_done_run >= a.limit:
                break
        if a.limit and n_done_run >= a.limit:
            break
        if done == len(rows):
            print("all videos in the selected lanes have features.", flush=True)
            break
        if progressed:
            last_progress = time.time()
        dl_running = downloading()
        idle_out = dl is None and not dl_running and time.time() - last_progress > a.idle_timeout
        if not a.wait or idle_out or (dl is not None and not dl_running and not progressed):
            print(f"stopping: {done}/{len(rows)} done, {pending} not available "
                  f"(not downloaded, failed, or claimed by another worker).", flush=True)
            break
        if not progressed:
            print(f"  waiting for downloads... {done}/{len(rows)} done, {pending} pending", flush=True)
            time.sleep(a.poll)

    el = time.time() - t_start
    if clips_run:
        print(f"this run: {n_done_run} videos, {clips_run} clips, {el / 60:.1f} min wall, "
              f"{1000 * model_secs / clips_run:.1f} ms/clip incl. decode", flush=True)
    if failures:
        print("failed videos:", sorted(failures), flush=True)
    if downloading():
        print("downloader still running; leaving it to finish.", flush=True)


if __name__ == "__main__":
    main()
