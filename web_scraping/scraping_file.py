from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from apify_client import ApifyClient
import yt_dlp

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "scraped_raw_data.json"
YOUTUBE_OUTPUT_FILE = OUTPUT_FILE
INSTAGRAM_OUTPUT_FILE = BASE_DIR / "scraped_instagram_data.json"
INSTAGRAM_URLS_FILE = BASE_DIR / "scraped_instagram_urls.json"

# 1. Define Multi-Source Targets (category, query)
# Compound boolean groups are split into individual phrase queries so YouTube's
# internal search parser handles exact phrases and exclusions reliably.
YOUTUBE_QUERIES: list[tuple[str, str]] = [
    ("burglary", 'ytsearch40:"cctv" "burglary" -movie -gta -prank'),
    ("burglary", 'ytsearch40:"cctv" "robbery" -movie -gta -prank'),
    ("burglary", 'ytsearch40:"cctv" "break in" -movie -gta -prank'),
    ("burglary", 'ytsearch40:"ring camera" "burglary" -skit -prank'),
    ("burglary", 'ytsearch40:"ring camera" "home invasion" -skit -prank'),
    ("fire", 'ytsearch80:"house fire" "raw footage" caught on camera -sims -game'),
    ("fire", 'ytsearch35:"structure fire" "first arriving"'),
    ("fire", 'ytsearch35:"structure fire" "helmet cam"'),
]

# Specific anomaly hashtags are ordered before generic security tags so
# deduplication prefers specific anomaly labels over generic ones.
INSTAGRAM_HASHTAGS: dict[str, str] = {
    "burglarycaughtoncamera": "burglary",
    "structurefire": "fire",
    "workingfire": "fire",
    "cctvfootage": "general_security",
    "ringcamera": "general_security",
}

# In apify/instagram-hashtag-scraper, resultsLimit applies PER hashtag
# (5 hashtags * 100 = up to 500 items per run).
INSTAGRAM_RESULTS_PER_HASHTAG = 100

SPECIFIC_CATEGORIES = {"burglary", "fire"}
NON_VIDEO_YT_PATH_SEGMENTS = ("/channel/", "/playlist", "/@", "/user/", "/c/")
LIVE_STATUSES = {"is_live", "is_upcoming", "post_live"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_duration(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return round(float(value), 2)
    return None


def _is_valid_youtube_video(entry: dict[str, Any]) -> bool:
    vid_id = entry.get("id")
    # Standard YouTube video & Shorts IDs are strictly 11 chars (channels=24, playlists=18-34).
    if not isinstance(vid_id, str) or len(vid_id) != 11:
        return False
    if entry.get("_type") in {"playlist", "multi_video"}:
        return False
    if entry.get("ie_key") in {"YoutubeTab", "YoutubePlaylist"}:
        return False
    if entry.get("is_live") is True or entry.get("live_status") in LIVE_STATUSES:
        return False

    raw_url = str(entry.get("url") or entry.get("webpage_url") or "")
    if any(seg in raw_url for seg in NON_VIDEO_YT_PATH_SEGMENTS):
        return False

    duration = entry.get("duration")
    if isinstance(duration, (int, float)) and duration <= 0:
        return False
    return True


def scrape_youtube() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scraped_at = _utc_now_iso()
    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for category, query in YOUTUBE_QUERIES:
            print(f"[YouTube] Scraping ({category}): {query}")
            try:
                res = ydl.extract_info(query, download=False)
            except Exception as exc:
                print(f"[YouTube] Warning: query failed ({query}): {exc}")
                continue

            if not res or "entries" not in res:
                continue

            for entry in res["entries"] or []:
                if not isinstance(entry, dict) or not _is_valid_youtube_video(entry):
                    continue

                vid_id = entry["id"]
                raw_url = entry.get("url") or entry.get("webpage_url") or ""
                if isinstance(raw_url, str) and raw_url.startswith(("http://", "https://")):
                    video_url = raw_url
                else:
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"

                records.append({
                    "uid": f"yt_{vid_id}",
                    "source": "YouTube",
                    "url": video_url,
                    "title": (entry.get("title") or "").strip(),
                    "duration_sec": _normalize_duration(entry.get("duration")),
                    "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
                    "category": category,
                    "query_or_tag": query,
                    "scraped_at": scraped_at,
                })
    return records


def _resolve_instagram_tag(item: dict[str, Any]) -> tuple[str, str]:
    input_url = str(item.get("inputUrl") or "").lower().rstrip("/")
    for tag, category in INSTAGRAM_HASHTAGS.items():
        if input_url.endswith(f"/{tag.lower()}"):
            return category, f"#{tag}"

    item_tags = {str(t).lstrip("#").lower() for t in (item.get("hashtags") or []) if t}
    for tag, category in INSTAGRAM_HASHTAGS.items():
        if tag.lower() in item_tags:
            return category, f"#{tag}"

    caption_lower = str(item.get("caption") or "").lower()
    for tag, category in INSTAGRAM_HASHTAGS.items():
        if f"#{tag.lower()}" in caption_lower:
            return category, f"#{tag}"

    return "unknown", "instagram_hashtag"


def scrape_instagram() -> list[dict[str, Any]]:
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise SystemExit("Set the APIFY_API_TOKEN environment variable to scrape Instagram.")
    client = ApifyClient(token)
    records: list[dict[str, Any]] = []
    scraped_at = _utc_now_iso()

    run_input = {
        "hashtags": list(INSTAGRAM_HASHTAGS.keys()),
        "resultsType": "reels",
        "resultsLimit": INSTAGRAM_RESULTS_PER_HASHTAG,
    }

    print("[Instagram] Running Apify Reels Scraper...")
    run = client.actor("apify/instagram-hashtag-scraper").call(run_input=run_input)
    if not run:
        print("[Instagram] Warning: Apify actor run returned None (possible timeout).")
        return []

    status = (
        run.get("status")
        if isinstance(run, dict)
        else getattr(run, "status", None)
    )
    if status != "SUCCEEDED":
        print(f"[Instagram] Warning: Apify actor finished with status={status!r}.")

    dataset_id = (
        (run.get("defaultDatasetId") or run.get("default_dataset_id"))
        if isinstance(run, dict)
        else (getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None))
    )

    dataset_client = None
    if dataset_id:
        dataset_client = client.dataset(dataset_id)
    elif hasattr(run, "id") and run.id:
        dataset_client = client.run(run.id).dataset()
    elif isinstance(run, dict) and run.get("id"):
        dataset_client = client.run(run["id"]).dataset()

    if not dataset_client:
        print("[Instagram] Error: Missing defaultDatasetId in Apify run response.")
        return []

    for item in dataset_client.iterate_items():
        if not isinstance(item, dict):
            continue

        code = item.get("shortCode") or item.get("id") or item.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        code = code.strip()

        item_type = item.get("type")
        if (
            item_type
            and item_type != "Video"
            and not item.get("videoUrl")
            and item.get("videoDuration") is None
        ):
            continue

        category, matched_tag = _resolve_instagram_tag(item)
        # Note: Instagram displayUrl is a signed CDN URL that expires after hours/days;
        # download thumbnails promptly if persistent offline access is required.
        records.append({
            "uid": f"ig_{code}",
            "source": "Instagram Reels",
            "url": item.get("url") or f"https://www.instagram.com/reel/{code}/",
            "title": (item.get("caption") or "").replace("\n", " ").strip()[:150],
            "duration_sec": _normalize_duration(item.get("videoDuration")),
            "thumbnail": item.get("displayUrl") or "",
            "category": category,
            "query_or_tag": matched_tag,
            "scraped_at": scraped_at,
        })
    return records


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    url_to_uid: dict[str, str] = {}

    for rec in records:
        uid = rec.get("uid")
        url = str(rec.get("url") or "").strip().rstrip("/")
        if not uid or not url:
            continue

        existing_uid = uid if uid in by_uid else url_to_uid.get(url)
        if existing_uid is not None:
            existing = by_uid[existing_uid]
            # Upgrade generic security label if a more specific anomaly query/hashtag also matched.
            if (
                existing.get("category") not in SPECIFIC_CATEGORIES
                and rec.get("category") in SPECIFIC_CATEGORIES
            ):
                existing["category"] = rec["category"]
                existing["query_or_tag"] = rec["query_or_tag"]
            continue

        by_uid[uid] = dict(rec)
        url_to_uid[url] = uid

    return list(by_uid.values())


def save_json_atomic(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, output_path)


def save_records_atomic(records: list[dict[str, Any]], output_path: Path) -> None:
    save_json_atomic(records, output_path)


def load_existing_records(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
    except Exception as exc:
        print(f"Warning: Could not read existing records from {file_path}: {exc}")
    return []


def main() -> None:
    print("Starting multi-source data scraping pipeline...")

    scrapers: list[tuple[str, Any, Path]] = [
        ("YouTube", scrape_youtube, YOUTUBE_OUTPUT_FILE),
        ("Instagram", scrape_instagram, INSTAGRAM_OUTPUT_FILE),
    ]

    for source_name, scraper_fn, output_file in scrapers:
        try:
            records = scraper_fn()
            print(f"[{source_name}] Collected {len(records)} raw records.")
            existing_records = load_existing_records(output_file)
            merged = deduplicate_records(existing_records + records)
            save_records_atomic(merged, output_file)
            print(
                f"[{source_name}] Successfully saved {len(merged)} unique records "
                f"to {output_file}."
            )

            # If Instagram, also output a dedicated URL-only list for convenience
            if source_name == "Instagram":
                urls_only = [r["url"] for r in merged if r.get("url")]
                save_json_atomic(urls_only, INSTAGRAM_URLS_FILE)
                print(
                    f"[{source_name}] Saved {len(urls_only)} URLs to {INSTAGRAM_URLS_FILE}."
                )
        except Exception as exc:
            print(f"[{source_name}] Error during scraping: {exc}")


if __name__ == "__main__":
    main()
