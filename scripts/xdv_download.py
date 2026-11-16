#!/usr/bin/env python3
"""Download XD-Violence VGGish (audio) + I3D RGB center-crop features to local/Drive dir.

Sources (verified public Kaggle datasets, no credentials required):
  - Audio (VGGish, 3,954 train + 800 test + annotations.txt + gt.npy):
      pawarlucky/kundan-xd-violence-audio-vggish  (0.24 GB zip / 0.62 GB raw) via kagglehub
  - Video (I3D RGB center crop __0.npy, 3,954 train + 800 test, range-fetched via remotezip):
      pawarlucky/kundan-xd-violence-i3d           (4.25 GB zip / 4.78 GB raw out of 41.26 GB)
      Resolves Kaggle's single 302 redirect to the direct GCS signed URL once so
      all range requests hit storage.googleapis.com directly without Kaggle rate limits.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

import requests

VGGISH_SLUG = "pawarlucky/kundan-xd-violence-audio-vggish"
I3D_SLUG = "pawarlucky/kundan-xd-violence-i3d"
HEADERS = {"User-Agent": "kagglehub/1.0.3"}


def download_vggish(out_dir: Path) -> None:
    """Download/link official VGGish audio features + annotations via kagglehub."""
    train_dir = out_dir / "vggish" / "train"
    test_dir = out_dir / "vggish" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    existing_train = len(list(train_dir.glob("*.npy")))
    existing_test = len(list(test_dir.glob("*.npy")))
    ann_path = out_dir / "annotations.txt"
    if existing_train >= 3954 and existing_test >= 800 and ann_path.exists():
        print(f"[xdv] VGGish already complete ({existing_train} train, {existing_test} test)", flush=True)
        return

    import kagglehub

    cache_root = Path(kagglehub.dataset_download(VGGISH_SLUG))
    print(f"[xdv] kagglehub VGGish cache: {cache_root}", flush=True)

    n_tr = n_te = 0
    for src_str in glob.glob(str(cache_root / "**" / "*.npy"), recursive=True):
        src = Path(src_str)
        low = src_str.lower()
        if "vggish" in low and "/train/" in low:
            dest = train_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            n_tr += 1
        elif "vggish" in low and "/test/" in low:
            dest = test_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            n_te += 1
        elif src.name == "gt.npy":
            shutil.copy2(src, out_dir / "gt.npy")

    for name in ("annotations.txt", "annotations_aligned.txt"):
        hits = glob.glob(str(cache_root / "**" / name), recursive=True)
        if hits:
            shutil.copy2(hits[0], out_dir / name)

    print(
        f"[xdv] VGGish ready: {n_tr} train, {n_te} test, annotations.txt={ann_path.exists()}",
        flush=True,
    )
    if n_tr < 3954 or n_te < 800:
        raise RuntimeError(f"Expected 3954 train / 800 test VGGish files, got {n_tr} / {n_te}")


def resolve_kaggle_gcs_url(slug: str) -> str:
    """Resolve Kaggle's single download redirect to a direct GCS signed URL."""
    view_url = f"https://www.kaggle.com/api/v1/datasets/view/{slug}"
    vr = requests.get(view_url, headers=HEADERS, timeout=20)
    vr.raise_for_status()
    ver = vr.json().get("currentVersionNumber") or 1
    dl_url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}?datasetVersionNumber={ver}"
    r = requests.get(
        dl_url,
        headers={**HEADERS, "Range": "bytes=0-0"},
        allow_redirects=True,
        stream=True,
        timeout=30,
    )
    r.raise_for_status()
    final_url = r.url
    r.close()
    print(f"[xdv] resolved {slug} (v{ver}) -> host={final_url.split('/')[2]}", flush=True)
    return final_url


def download_i3d_rgb_center(out_dir: Path, workers: int = 12, retries: int = 4) -> None:
    """Range-fetch ONLY official I3D RGB center-crop (__0.npy) directly from resolved GCS URL."""
    from remotezip import RemoteZip

    train_dir = out_dir / "i3d_rgb" / "train"
    test_dir = out_dir / "i3d_rgb" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    url_lock = threading.Lock()
    state = {"url": resolve_kaggle_gcs_url(I3D_SLUG)}

    def _refresh_url(old_url: str) -> str:
        with url_lock:
            if state["url"] == old_url:
                state["url"] = resolve_kaggle_gcs_url(I3D_SLUG)
            return state["url"]

    print("[xdv] reading I3D zip central directory...", flush=True)
    with RemoteZip(state["url"]) as rz:
        targets: list[tuple[str, Path, int]] = []
        for info in rz.infolist():
            fn = info.filename
            if "/RGB/" in fn and fn.endswith("__0.npy"):
                dest = train_dir / os.path.basename(fn)
                targets.append((fn, dest, info.file_size))
            elif "/RGBTest/" in fn and fn.endswith("__0.npy"):
                dest = test_dir / os.path.basename(fn)
                targets.append((fn, dest, info.file_size))

    pending = [(fn, dest, sz) for fn, dest, sz in targets if not (dest.exists() and dest.stat().st_size == sz)]
    done_initial = len(targets) - len(pending)
    total_gb = sum(sz for _, _, sz in pending) / 1e9
    print(
        f"[xdv] I3D RGB center-crop: {done_initial}/{len(targets)} already present, "
        f"{len(pending)} to fetch ({total_gb:.2f} GB raw) with {workers} workers",
        flush=True,
    )
    if not pending:
        return

    tls = threading.local()

    def _get_rz() -> tuple[RemoteZip, str]:
        cur = state["url"]
        rz = getattr(tls, "rz", None)
        rz_url = getattr(tls, "rz_url", None)
        if rz is None or rz_url != cur:
            if rz is not None:
                try:
                    rz.close()
                except Exception:
                    pass
            rz = RemoteZip(cur)
            tls.rz = rz
            tls.rz_url = cur
        return rz, cur

    def _fetch_one(item: tuple[str, Path, int]) -> tuple[str, int, str | None]:
        fn, dest, sz = item
        tmp = dest.with_suffix(".npy.tmp")
        for attempt in range(retries):
            cur_url = state["url"]
            try:
                rz, cur_url = _get_rz()
                data = rz.read(fn)
                if len(data) != sz:
                    raise IOError(f"short read {len(data)} != {sz}")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                return fn, sz, None
            except Exception as exc:
                old = getattr(tls, "rz", None)
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
                    tls.rz = None
                if "403" in str(exc) or "404" in str(exc):
                    try:
                        _refresh_url(cur_url)
                    except Exception:
                        pass
                if attempt + 1 == retries:
                    return fn, 0, f"{type(exc).__name__}: {exc}"
                time.sleep(1.0 * (2 ** attempt))
        return fn, 0, "max retries exceeded"

    t0 = time.time()
    fetched = done_initial
    bytes_done = 0
    failed: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for fn, sz, err in ex.map(_fetch_one, pending):
            if err is not None:
                failed.append((fn, err))
                print(f"  FAIL {os.path.basename(fn)}: {err}", flush=True)
            else:
                fetched += 1
                bytes_done += sz
                if fetched % 250 == 0 or fetched == len(targets):
                    dt = max(1e-3, time.time() - t0)
                    mbps = (bytes_done / 1e6) / dt
                    print(
                        f"  [{fetched:4d}/{len(targets)}] {bytes_done/1e9:.2f}/{total_gb:.2f} GB  "
                        f"({mbps:.1f} MB/s)",
                        flush=True,
                    )

    if failed:
        print(f"[xdv] WARNING: {len(failed)} I3D files failed; rerun to retry.", flush=True)
        raise SystemExit(1)
    print(f"[xdv] I3D RGB center-crop complete: {fetched}/{len(targets)} files.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/content/drive/MyDrive/vad/xdv", help="Output root directory")
    ap.add_argument("--workers", type=int, default=12, help="Parallel remotezip workers for I3D RGB")
    ap.add_argument("--audio-only", action="store_true", help="Download only VGGish audio + annotations (0.24 GB)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    download_vggish(out_dir)
    if not args.audio_only:
        download_i3d_rgb_center(out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
