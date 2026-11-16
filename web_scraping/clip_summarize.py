"""Download each scraped video URL one by one, summarise it with CLIP, save to a file.

For every record in the scraped JSON files this script:
  1. downloads the video with yt-dlp into a temp dir,
  2. samples N frames evenly across the video with ffmpeg,
  3. scores every frame against text prompts for each class with CLIP
     (zero-shot, as in Stage 2 of the proposal), including an "unrelated" class
     that catches video games, movies, cartoons, vlogs etc.,
  4. saves one frame as a thumbnail, appends one summary line to a JSONL file,
     deletes the video and refreshes a presentation-friendly CSV.

Resumable: uids already processed with the current PROMPT_VERSION are skipped, so
you can stop and re-run at any time. Records from an older prompt version are
re-processed automatically.

Usage:
    python clip_summarize.py                       # all URLs in both scraped files
    python clip_summarize.py --limit 5             # quick test
    python clip_summarize.py --csv-only            # rebuild the CSV, no downloads
    python clip_summarize.py --cookies-from-browser chrome   # Instagram often needs login
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yt_dlp
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    BASE_DIR / "scraped_raw_data.json",
    BASE_DIR / "scraped_instagram_data.json",
]
DEFAULT_OUTPUT = BASE_DIR / "clip_summaries.jsonl"
DEFAULT_CSV = BASE_DIR / "clip_summaries.csv"
DEFAULT_THUMB_DIR = BASE_DIR / "thumbnails"

# Bump whenever CLASS_PROMPTS or the decision rules change: records with an older
# version are re-processed on the next run.
PROMPT_VERSION = 2

# class -> text prompts (embeddings of a class's prompts are averaged)
CLASS_PROMPTS: dict[str, list[str]] = {
    "burglary": [
        "a surveillance camera video of a burglar breaking into a house",
        "a CCTV video of a thief stealing from a store",
        "a person forcing a door open to rob a home",
    ],
    "fire": [
        "a video of a building on fire with flames and smoke",
        "a house fire with fire fighters",
        "a CCTV video of a fire breaking out",
    ],
    "shooting": [
        "a surveillance video of a person firing a gun",
        "a shooting incident with people running away",
        "a person pointing a gun at someone",
    ],
    "fighting": [
        "a CCTV video of people fighting and punching each other",
        "a physical altercation between two people",
        "a street brawl caught on camera",
    ],
    "masked_person": [
        "a person wearing a mask hiding their face",
        "a masked intruder in a balaclava",
        "a person with a covered face on a security camera",
    ],
    "explosion": [
        "a video of a large explosion with a fireball",
        "an explosion and smoke cloud caught on camera",
    ],
    # real camera footage where nothing happens -> useful "normal" training data
    "normal": [
        "real CCTV footage of a quiet street with nothing unusual happening",
        "a real security camera view of people walking calmly",
        "real surveillance footage of an empty room, shop or parking lot",
    ],
    # not real-world camera footage -> must be kept out of the training data
    "unrelated": [
        "a screenshot of a video game",
        "gameplay footage from a video game like GTA",
        "a cartoon or animation",
        "a scene from a movie or TV show",
        "a news presenter talking in a studio",
        "a person talking to the camera in a vlog",
        "a title card or text on a plain background",
        "a meme with large text",
    ],
}
NON_ANOMALY_CLASSES = {"normal", "unrelated"}

# Separate real-vs-synthetic check. It does not compete with the anomaly classes,
# so a GTA shooting frame can't "win" as shooting. In tests the per-video median
# P(synthetic) was 0.75-0.96 for game/cartoon clips and <= 0.12 for real footage.
REALNESS_PROMPTS: dict[str, list[str]] = {
    "real": [
        "real CCTV security camera footage",
        "a real photo of a real place taken with a camera",
        "real smartphone video of a real event",
        "real news footage filmed on location",
        "a grainy real surveillance camera recording",
    ],
    "synthetic": CLASS_PROMPTS["unrelated"] + [
        "computer generated 3D graphics",
        "a video game screen with a minimap and on-screen HUD",
    ],
}
SYNTHETIC_MEDIAN_THRESHOLD = 0.5

# Backup check on the title/caption, in case CLIP misses an off-topic video.
UNRELATED_TITLE_RE = re.compile(
    r"\b(gta|gameplay|minecraft|roblox|fortnite|video ?game|animation|animated|cartoon|"
    r"trailer|movie scene|prank|skit)\b",
    re.IGNORECASE,
)

CSV_COLUMNS = [
    "uid", "platform", "url", "title", "duration_sec", "thumbnail_url", "thumbnail_file",
    "scraped_category", "query_or_tag", "status", "predicted_class", "confidence",
    "second_class", "second_prob", "third_class", "third_prob", "label_agrees",
    "is_unrelated", "unrelated_reason", "synthetic_score", "use_for_training", "needs_review",
    "peak_anomaly_class", "peak_anomaly_prob", "peak_anomaly_time_sec", "summary",
    "error", "prompt_version", "scraped_at",
]


# ----------------------------------------------------------------------------- IO
def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"[warn] input not found, skipping: {path}")
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            uid = rec.get("uid")
            if isinstance(rec, dict) and uid and rec.get("url") and uid not in seen:
                seen.add(uid)
                records.append(rec)
    return records


def load_summaries(output: Path) -> dict[str, dict[str, Any]]:
    """Latest summary per uid (later lines override earlier ones)."""
    latest: dict[str, dict[str, Any]] = {}
    if not output.exists():
        return latest
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            latest[row["uid"]] = row
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return latest


def platform_of(rec: dict[str, Any]) -> str:
    src = str(rec.get("source") or "").lower()
    url = str(rec.get("url") or "").lower()
    if "instagram" in src or "instagram.com" in url:
        return "Instagram Reel"
    if "youtube" in src or "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    return rec.get("source") or "unknown"


def write_csv(records: list[dict[str, Any]], summaries: dict[str, dict[str, Any]], path: Path) -> None:
    """One row per scraped record, joined with its latest CLIP summary (or 'pending')."""
    rows = []
    for rec in records:
        s = summaries.get(rec["uid"], {})
        top = s.get("top_classes") or []
        peak = s.get("peak_anomaly") or {}

        def top_at(i: int, key: str) -> Any:
            return top[i][key] if len(top) > i else ""

        rows.append({
            "uid": rec["uid"],
            "platform": platform_of(rec),
            "url": rec["url"],
            "title": rec.get("title", ""),
            "duration_sec": rec.get("duration_sec", ""),
            "thumbnail_url": rec.get("thumbnail", ""),
            "thumbnail_file": s.get("thumbnail_file", ""),
            "scraped_category": rec.get("category", ""),
            "query_or_tag": rec.get("query_or_tag", ""),
            "status": s.get("status", "pending"),
            "predicted_class": s.get("predicted_class", ""),
            "confidence": top_at(0, "mean_prob"),
            "second_class": top_at(1, "class"),
            "second_prob": top_at(1, "mean_prob"),
            "third_class": top_at(2, "class"),
            "third_prob": top_at(2, "mean_prob"),
            "label_agrees": s.get("label_agrees", ""),
            "is_unrelated": s.get("is_unrelated", ""),
            "unrelated_reason": s.get("unrelated_reason", ""),
            "synthetic_score": s.get("synthetic_score", ""),
            "use_for_training": s.get("use_for_training", ""),
            "needs_review": s.get("needs_review", ""),
            "peak_anomaly_class": peak.get("class", ""),
            "peak_anomaly_prob": peak.get("prob", ""),
            "peak_anomaly_time_sec": peak.get("timestamp_sec", ""),
            "summary": s.get("summary", ""),
            "error": s.get("error", ""),
            "prompt_version": s.get("prompt_version", 1 if s else ""),
            "scraped_at": rec.get("scraped_at", ""),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # utf-8-sig so Excel/Numbers show emoji and non-English captions correctly
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# ------------------------------------------------------------------ download/frames
def download_video(url: str, out_dir: Path, args: argparse.Namespace) -> Path:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_dir / "video.%(ext)s"),
        # small files are enough for CLIP (224px input)
        "format": f"best[height<={args.max_height}]/bestvideo[height<={args.max_height}]/best",
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in out_dir.iterdir() if p.name.startswith("video.")]
    if not files:
        raise RuntimeError("yt-dlp produced no file")
    return files[0]


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def extract_frames(video: Path, num_frames: int, out_dir: Path) -> list[tuple[float, Path]]:
    """Grab `num_frames` frames at the centres of equal-length segments."""
    duration = probe_duration(video)
    frames: list[tuple[float, Path]] = []
    for i in range(num_frames):
        ts = duration * (i + 0.5) / num_frames
        dest = out_dir / f"frame_{i:03d}.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{ts:.2f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(dest)],
            check=True,
        )
        if dest.exists():
            frames.append((round(ts, 2), dest))
    if not frames:
        raise RuntimeError("no frames extracted")
    return frames


# ------------------------------------------------------------------------- CLIP
class ClipScorer:
    def __init__(self, model_name: str):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[clip] loading {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.classes = list(CLASS_PROMPTS)
        self.text_emb = self._encode(CLASS_PROMPTS)
        self.real_emb = self._encode(REALNESS_PROMPTS)  # rows: real, synthetic

    # get_text_features/get_image_features return a tensor in some transformers
    # versions and a model-output object in others, so run encoder + projection
    # explicitly; the result is identical and version-independent.
    def _text_features(self, inputs) -> torch.Tensor:
        out = self.model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )
        return self.model.text_projection(out.pooler_output)

    def _image_features(self, inputs) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        return self.model.visual_projection(out.pooler_output)

    @torch.no_grad()
    def _encode(self, prompt_sets: dict[str, list[str]]) -> torch.Tensor:
        embs = []
        for prompts in prompt_sets.values():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            e = self._text_features(inputs)
            e = torch.nn.functional.normalize(e, dim=-1).mean(dim=0)
            embs.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.stack(embs)  # [C, D]

    @torch.no_grad()
    def score(self, frame_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (per-frame class probabilities [F, C], per-frame P(synthetic) [F])."""
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        img = torch.nn.functional.normalize(self._image_features(inputs), dim=-1)
        probs = (100.0 * img @ self.text_emb.T).softmax(dim=-1).cpu()
        p_synthetic = (100.0 * img @ self.real_emb.T).softmax(dim=-1)[:, 1].cpu()
        return probs, p_synthetic


def summarise(
    rec: dict[str, Any], frames: list[tuple[float, Path]], probs: torch.Tensor,
    p_synthetic: torch.Tensor, classes: list[str], top_k: int,
) -> tuple[dict[str, Any], int]:
    """Returns (summary record, index of the frame to use as thumbnail)."""
    mean = probs.mean(dim=0)
    order = mean.argsort(descending=True)[:top_k].tolist()
    top = [{"class": classes[i], "mean_prob": round(float(mean[i]), 4),
            "max_prob": round(float(probs[:, i].max()), 4)} for i in order]
    pred = classes[order[0]]

    # strongest anomaly evidence, so short anomalies in a long video are not averaged away
    anomaly_idx = [i for i, c in enumerate(classes) if c not in NON_ANOMALY_CLASSES]
    sub = probs[:, anomaly_idx]
    f_idx, c_sub = divmod(int(sub.argmax()), sub.shape[1])
    peak_cls = classes[anomaly_idx[c_sub]]
    peak_prob = float(sub[f_idx, c_sub])
    peak_ts = frames[f_idx][0]

    # unrelated: the real-vs-synthetic check, CLIP's top class, or a title keyword
    unrel = classes.index("unrelated")
    unrel_frame_share = float((probs.argmax(dim=1) == unrel).float().mean())
    synthetic_median = float(p_synthetic.median())
    title_hit = UNRELATED_TITLE_RE.search(str(rec.get("title") or ""))
    if synthetic_median >= SYNTHETIC_MEDIAN_THRESHOLD:
        reason = "clip_synthetic"
    elif pred == "unrelated":
        reason = "clip_top_class"
    elif title_hit:
        reason = f"title_keyword:{title_hit.group(0).lower()}"
    else:
        reason = ""
    is_unrelated = bool(reason)
    label_agrees = pred == rec.get("category")

    if is_unrelated:
        text = (f"Unrelated / not real footage ({reason}; synthetic score {synthetic_median:.0%}). "
                f"Excluded from training data.")
    else:
        text = (f"Most likely '{pred}' ({top[0]['mean_prob']:.0%} avg over {len(frames)} frames). "
                f"Strongest anomaly cue: '{peak_cls}' at ~{peak_ts}s ({peak_prob:.0%}).")
        if len(top) > 1:
            text += " Runner-up: " + ", ".join(f"{t['class']} {t['mean_prob']:.0%}" for t in top[1:]) + "."

    # thumbnail: the peak-anomaly frame for anomalies, otherwise the middle frame
    thumb_idx = f_idx if pred not in NON_ANOMALY_CLASSES and not is_unrelated else len(frames) // 2

    result = {
        "uid": rec["uid"],
        "url": rec["url"],
        "source": rec.get("source"),
        "platform": platform_of(rec),
        "title": rec.get("title"),
        "scraped_category": rec.get("category"),
        "status": "ok",
        "predicted_class": pred,
        "label_agrees": label_agrees,
        "is_unrelated": is_unrelated,
        "unrelated_reason": reason,
        "unrelated_frame_share": round(unrel_frame_share, 4),
        "synthetic_score": round(synthetic_median, 4),
        "use_for_training": not is_unrelated,
        "needs_review": not label_agrees,
        "top_classes": top,
        "peak_anomaly": {"class": peak_cls, "prob": round(peak_prob, 4), "timestamp_sec": peak_ts},
        "frame_timestamps": [t for t, _ in frames],
        "summary": text,
        "prompt_version": PROMPT_VERSION,
    }
    return result, thumb_idx


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="presentation CSV path")
    ap.add_argument("--thumb-dir", type=Path, default=DEFAULT_THUMB_DIR)
    ap.add_argument("--csv-only", action="store_true",
                    help="only rebuild the CSV from the existing JSONL, no downloads")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-height", type=int, default=480, help="max download resolution")
    ap.add_argument("--max-duration", type=float, default=600,
                    help="skip videos longer than this many seconds (default 600)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pending URLs")
    ap.add_argument("--cookies-from-browser", default=None, help="e.g. chrome, safari, firefox")
    ap.add_argument("--keep-videos", type=Path, default=None,
                    help="if set, move downloaded videos into this folder instead of deleting")
    args = ap.parse_args()

    records = load_records(args.input)
    summaries = load_summaries(args.output)

    if args.csv_only:
        write_csv(records, summaries, args.csv)
        print(f"[csv] wrote {len(records)} rows -> {args.csv}")
        return

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    finished = {uid for uid, s in summaries.items() if s.get("prompt_version") == PROMPT_VERSION}
    pending = [r for r in records if r["uid"] not in finished]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[info] {len(records)} records, {len(finished)} done with prompt v{PROMPT_VERSION}, "
          f"{len(pending)} to process")
    if not pending:
        write_csv(records, summaries, args.csv)
        return

    scorer = ClipScorer(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.thumb_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_videos:
        args.keep_videos.mkdir(parents=True, exist_ok=True)

    ok = failed = unrelated = 0
    try:
        with open(args.output, "a", encoding="utf-8") as out:
            for n, rec in enumerate(pending, 1):
                print(f"[{n}/{len(pending)}] {rec['uid']}  {rec['url']}")
                base = {"uid": rec["uid"], "url": rec["url"], "platform": platform_of(rec),
                        "prompt_version": PROMPT_VERSION}
                # known duration from scraping: avoid downloading huge videos
                dur = rec.get("duration_sec")
                if dur and dur > args.max_duration:
                    result = {**base, "status": "skipped_too_long", "duration_sec": dur}
                else:
                    with tempfile.TemporaryDirectory(prefix="clipvid_") as tmp:
                        tmp_dir = Path(tmp)
                        try:
                            video = download_video(rec["url"], tmp_dir, args)
                            frames = extract_frames(video, args.num_frames, tmp_dir)
                            probs, p_synth = scorer.score([p for _, p in frames])
                            result, thumb_idx = summarise(rec, frames, probs, p_synth, scorer.classes, args.top_k)
                            thumb = args.thumb_dir / f"{rec['uid']}.jpg"
                            shutil.copyfile(frames[thumb_idx][1], thumb)
                            try:
                                result["thumbnail_file"] = str(thumb.relative_to(args.csv.parent))
                            except ValueError:
                                result["thumbnail_file"] = str(thumb)
                            if args.keep_videos:
                                shutil.move(str(video), args.keep_videos / f"{rec['uid']}{video.suffix}")
                            print(f"    -> {result['summary']}")
                            ok += 1
                            unrelated += result["is_unrelated"]
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:  # keep going; record the failure
                            print(f"    !! failed: {exc}")
                            result = {**base, "status": "failed", "error": str(exc)[:300]}
                            failed += 1
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                summaries[rec["uid"]] = result
                write_csv(records, summaries, args.csv)
    finally:
        write_csv(records, summaries, args.csv)
        print(f"[done] ok={ok} (unrelated={unrelated}) failed={failed} -> {args.output}, {args.csv}")


if __name__ == "__main__":
    main()
