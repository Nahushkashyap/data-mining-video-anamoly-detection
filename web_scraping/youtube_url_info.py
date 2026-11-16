"""Print a scraped_raw_data.json-style record for one or more YouTube URLs.

Nothing is written to any file: the JSON is printed so you can verify it and
copy-paste it into scraped_raw_data.json yourself.

Usage:
    python youtube_url_info.py https://www.youtube.com/watch?v=CAL2VaT-0JI
    python youtube_url_info.py URL1 URL2 --category fire
    python youtube_url_info.py            # prompts for URLs, one per line
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yt_dlp

EXISTING_FILE = Path(__file__).resolve().parent / "scraped_raw_data.json"


def fetch_record(url: str, category: str, query_or_tag: str) -> dict[str, Any]:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    vid_id = info["id"]
    duration = info.get("duration")
    return {
        "uid": f"yt_{vid_id}",
        "source": "YouTube",
        "url": f"https://www.youtube.com/watch?v={vid_id}",
        "title": (info.get("title") or "").strip(),
        "duration_sec": round(float(duration), 2) if isinstance(duration, (int, float)) and duration > 0 else None,
        "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
        "category": category,
        "query_or_tag": query_or_tag,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def existing_uids() -> set[str]:
    try:
        return {r.get("uid") for r in json.loads(EXISTING_FILE.read_text(encoding="utf-8"))}
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="YouTube video or Shorts URLs")
    ap.add_argument("--category", default="",
                    help="burglary, fire, shooting, fighting, masked_person, ... (left empty if not given)")
    ap.add_argument("--query-or-tag", default="manual", help='provenance note (default "manual")')
    args = ap.parse_args()

    urls = args.urls
    if not urls:
        print("Paste YouTube URLs, one per line. Empty line to finish:", file=sys.stderr)
        urls = [line.strip() for line in iter(input, "") if line.strip()]

    known = existing_uids()
    records = []
    for url in urls:
        try:
            rec = fetch_record(url, args.category, args.query_or_tag)
        except Exception as exc:
            print(f"[error] {url}: {exc}", file=sys.stderr)
            continue
        if rec["uid"] in known:
            print(f"[note] {rec['uid']} is already in {EXISTING_FILE.name}", file=sys.stderr)
        if not rec["category"]:
            print(f"[note] {rec['uid']}: category is empty, fill it in before pasting", file=sys.stderr)
        records.append(rec)

    # Comma-separated objects, ready to paste inside the existing JSON list.
    print(",\n".join(json.dumps(r, indent=2, ensure_ascii=False) for r in records))


if __name__ == "__main__":
    main()
