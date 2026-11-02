import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import kagglehub

# Dataset configuration
DATASET_HANDLE = "webadvisor/real-time-anomaly-detection-in-cctv-surveillance"

# Resolve path to web_scraping/dataset (where existing categories reside)
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"

# Target categories to download
FOLDERS = ["explosion", "burglary", "stealing", "normal"]


def unnest_folder(category: str) -> None:
    """Flatten any nested 'data/{category}' directory left from prior runs."""
    target_dir = DATASET_DIR / category
    nested_dir = target_dir / "data" / category
    if nested_dir.exists():
        print(f"[CLEANUP] Un-nesting leftover directory for '{category}'...")
        for item in nested_dir.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                try:
                    shutil.move(str(item), str(dest))
                except Exception:
                    pass
            else:
                try:
                    if item.is_file():
                        item.unlink()
                except Exception:
                    pass
        shutil.rmtree(target_dir / "data", ignore_errors=True)


def download_folder(category: str) -> str:
    """
    Download a single category from Kaggle into DATASET_DIR/<category>.
    Skips if video files already exist.
    """
    target_dir = DATASET_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean up any nested directories if this category was partially downloaded earlier
    unnest_folder(category)

    # 2. Check if category already contains videos (either direct or nested)
    existing_videos = [
        f for f in target_dir.iterdir() if f.suffix.lower() in {".mp4", ".avi", ".mkv"}
    ]
    if not existing_videos and (target_dir / "data" / category).exists():
        existing_videos = [
            f for f in (target_dir / "data" / category).iterdir()
            if f.suffix.lower() in {".mp4", ".avi", ".mkv"}
        ]

    if existing_videos:
        print(f"[SKIP] '{category}' already has {len(existing_videos)} videos in {target_dir}")
        return str(target_dir)

    print(f"[START] Downloading '{category}'...")
    try:
        # kagglehub downloads to DATASET_DIR/data/<category>
        raw_download_path = kagglehub.dataset_download(
            DATASET_HANDLE,
            path=f"data/{category}",
            output_dir=str(DATASET_DIR),
            force_download=False,
        )
        downloaded_dir = Path(raw_download_path)

        # 3. Move files from DATASET_DIR/data/<category> to DATASET_DIR/<category>
        if downloaded_dir.exists() and downloaded_dir != target_dir:
            print(f"[EXTRACTING] Moving files for '{category}' to {target_dir}...")
            for item in downloaded_dir.iterdir():
                dest = target_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                else:
                    if item.is_file():
                        item.unlink()

            # Clean up the downloaded source folder
            shutil.rmtree(downloaded_dir, ignore_errors=True)

        # Try to remove empty parent 'data' folder inside DATASET_DIR
        try:
            (DATASET_DIR / "data").rmdir()
        except OSError:
            pass

        print(f"[SUCCESS] '{category}' ready -> {target_dir}")
        return str(target_dir)

    except Exception as e:
        print(f"[ERROR] Failed to download '{category}': {e}")
        raise e


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-clean existing folders in case any were left nested
    for cat in FOLDERS:
        unnest_folder(cat)

    # Number of concurrent workers (2-3 recommended to avoid connection throttling)
    max_workers = 3
    print(f"Starting parallel download for: {FOLDERS} (Workers: {max_workers})\n")

    results: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {
            executor.submit(download_folder, cat): cat for cat in FOLDERS
        }

        for future in as_completed(future_to_folder):
            cat = future_to_folder[future]
            try:
                path = future.result()
                results[cat] = ("SUCCESS", path)
            except Exception as e:
                results[cat] = ("FAILED", str(e))

    # Final cleanup of the intermediate 'data' directory if empty
    try:
        (DATASET_DIR / "data").rmdir()
    except OSError:
        pass

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for cat, (status, info) in results.items():
        print(f"[{status}] {cat}: {info}")


if __name__ == "__main__":
    main()
