#!/usr/bin/env python3
"""Download the UCF-Crime subset used by this project, one video at a time.

Videos are pulled straight out of the Hugging Face mirror's zips with HTTP range
requests (remotezip), so no full zip is ever downloaded.

Lanes (see "Execution Plan.pdf"):
  L1  Mac    test anomaly (76) + test normal (150) + train anomaly (374) = 600 videos, ~17.6 GB
  L2  Colab  400 train normals, seeded random sample                       ~37 GB
  L2X Colab  the other 400 train normals (stretch goal)                     ~37 GB

The manifest is deterministic (fixed seed), so the Mac and Colab agree on which
videos belong to which lane without talking to each other.

Re-running is safe: finished files (correct size) are skipped and partial files
are restarted. Only one downloader runs per output folder (.downloader.lock);
a second one exits immediately with code 3. An optional Hugging Face token can be set in the HF_TOKEN
variable below (or the HF_TOKEN environment variable).

Mac usage (download only):
  caffeinate -i venv/bin/python scripts/ucf_download.py --lanes L1 --workers 8
Download + process together: use scripts/extract_x3d.py --with-download (it starts this script).
"""
import argparse
import csv
import os
import random
import shutil
import socket
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from remotezip import RemoteZip

# Optional Hugging Face token (helps only if you hit HTTP 429 rate limits).
# Paste it between the quotes before running; leave empty for anonymous access.
# Don't share or commit this file with the token filled in.
HF_TOKEN = ""  # or export HF_TOKEN=... (never commit a real token)

BASE = "https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/"
ZIPS = [
    "Anomaly-Videos-Part-1.zip",
    "Anomaly-Videos-Part-2.zip",
    "Anomaly-Videos-Part-3.zip",
    "Anomaly-Videos-Part-4.zip",
    "Testing_Normal_Videos.zip",
    "Training-Normal-Videos-Part-1.zip",
    "Training-Normal-Videos-Part-2.zip",
]
SPLIT_FILES = {
    "Anomaly_Train.txt": "UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Train.txt",
    "Anomaly_Test.txt": "UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt",
    "Temporal_Anomaly_Annotation_for_Testing_Videos.txt":
        "UCF_Crimes-Train-Test-Split/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
}
CLASSES = {"Arson", "Burglary", "Explosion", "Fighting", "Robbery", "Shooting"}
SEED = 415
N_TRAIN_NORMAL = 400
MANIFEST_FIELDS = ["video_id", "split", "label", "cls", "lane", "zip", "member", "bytes", "rel_path"]


def hf_headers():
    token = (HF_TOKEN or os.environ.get("HF_TOKEN", "")).strip()
    h = {"User-Agent": "ucf-subset-downloader (research)"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def ensure_splits(split_dir):
    os.makedirs(split_dir, exist_ok=True)
    for name, remote in SPLIT_FILES.items():
        path = os.path.join(split_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        req = urllib.request.Request(BASE + remote, headers=hf_headers())
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
        print(f"fetched {name}")


def zip_index():
    """basename -> (zip, member, size) for every mp4 in the relevant zips."""
    idx = {}
    for z in ZIPS:
        with RemoteZip(BASE + z, headers=hf_headers()) as rz:
            for info in rz.infolist():
                if not info.filename.endswith(".mp4"):
                    continue
                b = os.path.basename(info.filename)
                if b in idx:
                    raise RuntimeError(f"duplicate basename {b} in {z} and {idx[b][0]}")
                idx[b] = (z, info.filename, info.file_size)
    return idx


def build_manifest(split_dir, manifest_path):
    ensure_splits(split_dir)
    idx = zip_index()
    rows, train_normals = [], []
    for split, fname in (("train", "Anomaly_Train.txt"), ("test", "Anomaly_Test.txt")):
        for line in open(os.path.join(split_dir, fname)):
            p = line.strip()
            if not p:
                continue
            folder, b = p.split("/")[0], os.path.basename(p)
            if folder in CLASSES:
                cls, label = folder, 1
            elif "Normal" in folder:
                cls, label = "Normal", 0
            else:
                continue
            z, member, size = idx[b]
            row = {
                "video_id": b[:-4], "split": split, "label": label, "cls": cls, "lane": "",
                "zip": z, "member": member, "bytes": size,
                "rel_path": f"{split.capitalize()}/{cls}/{b}",
            }
            if split == "train" and label == 0:
                train_normals.append(row)
            else:
                row["lane"] = "L1"
                rows.append(row)
    train_normals.sort(key=lambda r: r["video_id"])
    chosen = set(r["video_id"] for r in random.Random(SEED).sample(train_normals, N_TRAIN_NORMAL))
    for r in train_normals:
        r["lane"] = "L2" if r["video_id"] in chosen else "L2X"
        rows.append(r)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_or_build_manifest(split_dir, manifest_path):
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["bytes"] = int(r["bytes"])
            r["label"] = int(r["label"])
        return rows
    print("building manifest (lists the 7 remote zips, ~30 s)...")
    return build_manifest(split_dir, manifest_path)


_local = threading.local()


def _zip_handle(z):
    cache = getattr(_local, "zips", None)
    if cache is None:
        cache = _local.zips = {}
    if z not in cache:
        cache[z] = RemoteZip(BASE + z, headers=hf_headers())
    return cache[z]


def _drop_handle(z):
    cache = getattr(_local, "zips", {})
    rz = cache.pop(z, None)
    if rz is not None:
        try:
            rz.close()
        except Exception:
            pass


def download_one(row, out_root, retries=6):
    dest = os.path.join(out_root, row["rel_path"])
    if os.path.exists(dest) and os.path.getsize(dest) == row["bytes"]:
        return "skip", 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            rz = _zip_handle(row["zip"])
            with rz.open(row["member"]) as src, open(tmp, "wb") as dst:  # zipfile checks the CRC at EOF
                shutil.copyfileobj(src, dst, 4 << 20)
            got = os.path.getsize(tmp)
            if got != row["bytes"]:
                raise IOError(f"size mismatch {got} != {row['bytes']}")
            os.replace(tmp, dest)
            return "ok", row["bytes"]
        except Exception as e:  # network hiccup, 429, CRC error...
            _drop_handle(row["zip"])
            wait = min(120, 5 * 2 ** attempt)
            print(f"  retry {attempt + 1}/{retries} {row['video_id']}: {type(e).__name__}: {e} (sleep {wait}s)",
                  flush=True)
            time.sleep(wait)
    if os.path.exists(tmp):
        os.remove(tmp)
    return "fail", 0


def run(rows, out_root, workers, log_path=None, check_disk=True):
    todo = [r for r in rows
            if not (os.path.exists(os.path.join(out_root, r["rel_path"]))
                    and os.path.getsize(os.path.join(out_root, r["rel_path"])) == r["bytes"])]
    total_b = sum(r["bytes"] for r in todo)
    print(f"{len(rows)} videos in selection, {len(rows) - len(todo)} already done, "
          f"{len(todo)} to fetch ({total_b / 1e9:.2f} GB) with {workers} workers", flush=True)
    if not todo:
        return []
    free = shutil.disk_usage(out_root if os.path.exists(out_root) else ".").free
    if check_disk and free < total_b + 2e9:
        sys.exit(f"not enough disk: need {total_b / 1e9:.1f} GB + 2 GB headroom, have {free / 1e9:.1f} GB")
    t0, done_b, n_done, failed = time.time(), 0, 0, []
    log = open(log_path, "a") if log_path else None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, r, out_root): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            status, nb = fut.result()
            n_done += 1
            done_b += nb
            if status == "fail":
                failed.append(r["video_id"])
            el = time.time() - t0
            rate = done_b / el if el else 0
            eta = (total_b - done_b) / rate / 60 if rate else float("nan")
            msg = (f"[{n_done}/{len(todo)}] {status:4s} {r['rel_path']} {r['bytes'] / 1e6:.0f} MB | "
                   f"{done_b / 1e9:.2f}/{total_b / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s  ETA {eta:.0f} min")
            print(msg, flush=True)
            if log:
                log.write(msg + "\n")
                log.flush()
    print(f"finished in {(time.time() - t0) / 60:.1f} min; failed: {len(failed)} {failed[:10]}", flush=True)
    return failed


LOCK_NAME = ".downloader.lock"
LOCK_FRESH = 120  # seconds; the holder touches the lock every 30 s
EXIT_ALREADY_RUNNING = 3


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def lock_is_fresh(out_root):
    p = os.path.join(out_root, LOCK_NAME)
    try:
        return time.time() - os.path.getmtime(p) < LOCK_FRESH
    except OSError:
        return False


def acquire_lock(out_root):
    """One downloader per output folder. Returns False if another live one holds it."""
    p = os.path.join(out_root, LOCK_NAME)
    host = socket.gethostname()
    if lock_is_fresh(out_root):
        try:
            h, pid = open(p).read().split()[:2]
            if h != host or _pid_alive(int(pid)):
                return False
        except Exception:
            pass
    with open(p, "w") as f:
        f.write(f"{host} {os.getpid()}\n")

    def beat():
        while True:
            time.sleep(30)
            try:
                os.utime(p)
            except OSError:
                pass
    threading.Thread(target=beat, daemon=True).start()
    return True


def release_lock(out_root):
    p = os.path.join(out_root, LOCK_NAME)
    try:
        if open(p).read().split()[1] == str(os.getpid()):
            os.remove(p)
    except Exception:
        pass


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", default="L1", help="comma list of L1,L2,L2X")
    ap.add_argument("--out", default=os.path.join(proj, "ucf_videos.nosync"), help="video output root")
    ap.add_argument("--splits", default=os.path.join(proj, "data", "splits"))
    ap.add_argument("--manifest", default=os.path.join(proj, "data", "manifests", "ucf_subset.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (testing)")
    ap.add_argument("--no-disk-check", action="store_true", help="skip the free-space check (use on a Drive mount)")
    a = ap.parse_args()
    rows = load_or_build_manifest(a.splits, a.manifest)
    lanes = set(a.lanes.split(","))
    sel = [r for r in rows if r["lane"] in lanes]
    order = {"test": 0, "train": 1}
    sel.sort(key=lambda r: (order[r["split"]], -r["label"], r["video_id"]))  # eval set first
    if a.limit:
        sel = sel[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    if not acquire_lock(a.out):
        print(f"another downloader is already running on {a.out} (see {LOCK_NAME}); not starting a second one.",
              flush=True)
        sys.exit(EXIT_ALREADY_RUNNING)
    try:
        failed = run(sel, a.out, a.workers, log_path=os.path.join(a.out, "download_log.txt"),
                     check_disk=not a.no_disk_check)
    finally:
        release_lock(a.out)
    sys.exit(1 if failed else 0)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
