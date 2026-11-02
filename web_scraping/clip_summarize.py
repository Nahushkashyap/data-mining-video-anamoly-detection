"""Download each scraped video URL one by one, summarise it with CLIP, save to a file.

For every record in the scraped JSON files this script:
  1. downloads the video with yt-dlp into a temp dir,
  2. samples N frames evenly across the video with ffmpeg,
  3. scores every frame against text prompts for each anomaly class with CLIP
     (zero-shot, as in Stage 2 of the proposal),
  4. appends one summary line to a JSONL file and deletes the video.

Resumable: uids already present in the output file are skipped, so you can stop
and re-run at any time.

Usage:
    python clip_summarize.py                       # all URLs in both scraped files
    python clip_summarize.py --limit 5             # quick test
    python clip_summarize.py --cookies-from-browser chrome   # Instagram often needs login
"""
from __future__ import annotations

import argparse
import json
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
    "normal": [
        "a normal surveillance video with nothing unusual happening",
        "people walking calmly on a street",
        "an empty room or street seen by a security camera",
    ],
}


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


def done_uids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    uids = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            uids.add(json.loads(line)["uid"])
        except (json.JSONDecodeError, KeyError):
            continue
    return uids


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
        self.text_emb = self._encode_classes()

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
    def _encode_classes(self) -> torch.Tensor:
        embs = []
        for cls in self.classes:
            inputs = self.processor(text=CLASS_PROMPTS[cls], return_tensors="pt", padding=True).to(self.device)
            e = self._text_features(inputs)
            e = torch.nn.functional.normalize(e, dim=-1).mean(dim=0)
            embs.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.stack(embs)  # [C, D]

    @torch.no_grad()
    def score(self, frame_paths: list[Path]) -> torch.Tensor:
        """Returns per-frame class probabilities, shape [F, C]."""
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        img = torch.nn.functional.normalize(self._image_features(inputs), dim=-1)
        logits = 100.0 * img @ self.text_emb.T
        return logits.softmax(dim=-1).cpu()


def summarise(
    rec: dict[str, Any], frames: list[tuple[float, Path]], probs: torch.Tensor,
    classes: list[str], top_k: int,
) -> dict[str, Any]:
    mean = probs.mean(dim=0)
    order = mean.argsort(descending=True)[:top_k].tolist()
    top = [{"class": classes[i], "mean_prob": round(float(mean[i]), 4),
            "max_prob": round(float(probs[:, i].max()), 4)} for i in order]
    pred = classes[order[0]]

    # strongest non-normal evidence, so short anomalies in a long video are not averaged away
    anomaly_idx = [i for i, c in enumerate(classes) if c != "normal"]
    sub = probs[:, anomaly_idx]
    f_idx, c_sub = divmod(int(sub.argmax()), sub.shape[1])
    peak_cls = classes[anomaly_idx[c_sub]]
    peak_prob = float(sub[f_idx, c_sub])
    peak_ts = frames[f_idx][0]

    text = (
        f"Most likely '{pred}' ({top[0]['mean_prob']:.0%} avg over {len(frames)} frames). "
        f"Strongest anomaly cue: '{peak_cls}' at ~{peak_ts}s ({peak_prob:.0%})."
    )
    if len(top) > 1:
        text += " Runner-up: " + ", ".join(f"{t['class']} {t['mean_prob']:.0%}" for t in top[1:]) + "."

    return {
        "uid": rec["uid"],
        "url": rec["url"],
        "source": rec.get("source"),
        "title": rec.get("title"),
        "scraped_category": rec.get("category"),
        "status": "ok",
        "predicted_class": pred,
        "label_agrees": pred == rec.get("category"),
        "top_classes": top,
        "peak_anomaly": {"class": peak_cls, "prob": round(peak_prob, 4), "timestamp_sec": peak_ts},
        "frame_timestamps": [t for t, _ in frames],
        "summary": text,
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    records = load_records(args.input)
    finished = done_uids(args.output)
    pending = [r for r in records if r["uid"] not in finished]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[info] {len(records)} records, {len(finished)} already done, {len(pending)} to process")
    if not pending:
        return

    scorer = ClipScorer(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_videos:
        args.keep_videos.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for n, rec in enumerate(pending, 1):
            print(f"[{n}/{len(pending)}] {rec['uid']}  {rec['url']}")
            # known duration from scraping: avoid downloading huge videos
            dur = rec.get("duration_sec")
            if dur and dur > args.max_duration:
                result = {"uid": rec["uid"], "url": rec["url"], "status": "skipped_too_long",
                          "duration_sec": dur}
            else:
                with tempfile.TemporaryDirectory(prefix="clipvid_") as tmp:
                    tmp_dir = Path(tmp)
                    try:
                        video = download_video(rec["url"], tmp_dir, args)
                        frames = extract_frames(video, args.num_frames, tmp_dir)
                        probs = scorer.score([p for _, p in frames])
                        result = summarise(rec, frames, probs, scorer.classes, args.top_k)
                        if args.keep_videos:
                            shutil.move(str(video), args.keep_videos / f"{rec['uid']}{video.suffix}")
                        print(f"    -> {result['summary']}")
                        ok += 1
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # keep going; record the failure
                        print(f"    !! failed: {exc}")
                        result = {"uid": rec["uid"], "url": rec["url"], "status": "failed",
                                  "error": str(exc)[:300]}
                        failed += 1
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    print(f"[done] ok={ok} failed={failed} -> {args.output}")


if __name__ == "__main__":
    main()
