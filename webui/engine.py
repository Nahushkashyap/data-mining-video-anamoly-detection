"""Model engine for the web UI: X3D-S -> Stage 1 MIL head -> τ gate -> Stage 2 CLIP.

Uses the exact project pipeline pieces (scripts/extract_x3d.py preprocessing,
vad.pipeline.run_video windowing and frame sampling, vad.stage2.clip_scorer), so
uploaded-video results match `python -m vad.pipeline.run_video` on the same file.

All model files stay inside the project folder (`models_cache/`, `checkpoints/`):
the env vars below are set before torch.hub / transformers are touched.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "models_cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "models_cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import base64  # noqa: E402
import io  # noqa: E402
import queue  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from typing import Any, Callable  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.extract_x3d import CLIP_LEN, MEAN, STD, STRIDE, _prep, iter_clip_batches, load_x3d  # noqa: E402
from vad.data.ucf_dataset import l2_normalise  # noqa: E402
from vad.eval.metrics import smooth_scores  # noqa: E402
from vad.pipeline.run_video import extract_flagged_windows, load_stage1_checkpoint, sample_window_frames  # noqa: E402
from vad.stage2.clip_scorer import DEFAULT_SUPPRESS_THRESHOLD, ClipScorer, pick_device  # noqa: E402
from vad.audio.audio_io import load_audio  # noqa: E402
from vad.audio.clap_scorer import AUDIO_TO_VIDEO_CLASS, CLAP_SR  # noqa: E402
from vad.audio.pipeline import audio_windows, fuse  # noqa: E402
from vad.audio.trigger import DEFAULT_AUDIO_TAU, EAT_SR  # noqa: E402

FRAMES_PER_CLIP = CLIP_LEN * STRIDE  # 78 raw frames per clip
DATA_DIR = ROOT / "webui" / "data"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best_seed416.pt"


def _jpeg_data_url(img: Image.Image, max_w: int = 480, quality: int = 82) -> str:
    im = img.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _event_fields(res: dict[str, Any]) -> dict[str, Any]:
    # Design rule (§3.4): a window is an alert unless P(normal)+P(unrelated) >= the veto threshold. When CLIP's
    # top-1 is itself 'normal'/'unrelated' but below the veto, label the alert with the most likely *anomaly*
    # class (the Anomaly-7 top-1 used in the Stage 2 evaluation) instead of showing "⚠ NORMAL" (22 Nov).
    cls, conf = res["class"], res["confidence"]
    if not res["suppressed"] and cls in ("normal", "unrelated") and res.get("top3_anomaly"):
        cls, conf = res["top3_anomaly"][0]["class"], res["top3_anomaly"][0]["prob"]
    return {
        "class": cls,
        "confidence": conf,
        "clip_top_class": res["class"],
        "top3": res["top3"],
        "top3_anomaly": res["top3_anomaly"],
        "mean_probs": res["mean_probs"],
        "suppressed": bool(res["suppressed"]),
        "suppress_reason": res["suppress_reason"],
        "non_anomaly_prob": res["non_anomaly_prob"],
        "summary": res["summary"],
        **({"details": res["details"]} if "details" in res else {}),
    }


class Engine:
    """Loads the three models once and serialises GPU/MPS work behind one lock."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "auto") -> None:
        self.checkpoint = Path(checkpoint or os.environ.get("VAD_CHECKPOINT") or DEFAULT_CHECKPOINT)
        self.device_name = pick_device(os.environ.get("VAD_DEVICE", device))
        self.lock = threading.RLock()
        self.state = "idle"  # idle | loading | ready | error
        self.error = ""
        self.tau = None
        self.smooth_window = 3
        self.ckpt_meta: dict[str, Any] = {}
        self.audio_state = "idle"  # idle | loading | ready | error
        self.audio_error = ""
        self.tau_a = DEFAULT_AUDIO_TAU

    # ------------------------------------------------------------------ loading
    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error": self.error,
            "device": self.device_name,
            "checkpoint": str(self.checkpoint.relative_to(ROOT)) if self.checkpoint.is_relative_to(ROOT) else str(self.checkpoint),
            "checkpoint_exists": self.checkpoint.exists(),
            "tau": self.tau,
            "suppress_threshold": DEFAULT_SUPPRESS_THRESHOLD,
            "smooth_window": self.smooth_window,
            "holdout_auc": self.ckpt_meta.get("holdout_video_auc"),
            "seed": self.ckpt_meta.get("seed"),
            "clip_sec_at_30fps": FRAMES_PER_CLIP / 30.0,
            "audio_state": self.audio_state,
            "audio_error": self.audio_error,
            "tau_a": self.tau_a,
            "audio_model": getattr(getattr(self, "trigger", None), "mode", None),
        }

    def load(self) -> None:
        with self.lock:
            if self.state == "ready":
                return
            self.state, self.error = "loading", ""
            try:
                if not self.checkpoint.exists():
                    raise FileNotFoundError(
                        f"Stage 1 checkpoint not found at {self.checkpoint}. Copy best_seed416.pt from "
                        "MyDrive/vad/runs/stage1_sultani/ into the project's checkpoints/ folder."
                    )
                dev = torch.device(self.device_name)
                t0 = time.time()
                self.x3d, self.x3d_dtype = load_x3d(dev)
                self.mil, ckpt = load_stage1_checkpoint(self.checkpoint, device=dev)
                self.tau = float(ckpt["tau_info"]["tau"])
                self.smooth_window = int(ckpt.get("smooth_window", 3))
                self.ckpt_meta = {k: ckpt.get(k) for k in ("seed", "best_step", "holdout_video_auc")}
                self.clip = ClipScorer(device=self.device_name)
                self._mean = torch.tensor(MEAN, device=dev, dtype=self.x3d_dtype)
                self._std = torch.tensor(STD, device=dev, dtype=self.x3d_dtype)
                # warm-up so the first real request isn't slow
                self.embed(np.zeros((1, CLIP_LEN, 182, 182, 3), dtype=np.uint8))
                self.clip.score_window([Image.new("RGB", (224, 224))])
                print(f"[engine] models ready on {self.device_name} in {time.time() - t0:.1f}s (tau={self.tau:.4f})", flush=True)
                self.state = "ready"
            except Exception as exc:  # surfaced in /api/status
                traceback.print_exc()
                self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
        if self.state == "ready" and self.audio_state in ("idle", "error"):
            self.load_audio()

    def load_audio(self) -> None:
        """Audio Stage 1 (EfficientAT) + Stage 2 (MS-CLAP 2023). Loaded after video so the UI is usable sooner."""
        self.audio_state, self.audio_error = "loading", ""
        try:
            from vad.audio.clap_scorer import ClapScorer
            from vad.audio.trigger import AudioTrigger
            t0 = time.time()
            trig = AudioTrigger(device=self.device_name)
            clap = ClapScorer(device=self.device_name)
            with self.lock:
                self.trigger, self.clap = trig, clap
                self.trigger.score(np.zeros(EAT_SR, np.float32))
                self.clap.score_segment(np.zeros(CLAP_SR, np.float32), CLAP_SR)
            print(f"[engine] audio models ready in {time.time() - t0:.1f}s (model={trig.mode}, tau_a={self.tau_a})", flush=True)
            self.audio_state = "ready"
        except Exception as exc:
            traceback.print_exc()
            self.audio_state, self.audio_error = "error", f"{type(exc).__name__}: {exc}"

    def audio_score(self, wav32: np.ndarray):
        with self.lock:
            return self.trigger.score(wav32)

    def clap_segment(self, wav: np.ndarray, sr: int, suppress: float, details: bool = True) -> dict[str, Any]:
        with self.lock:
            return self.clap.score_segment(wav, sr, suppress_threshold=suppress, details=details)

    def require_ready(self) -> None:
        if self.state != "ready":
            raise RuntimeError(self.error or f"models are {self.state}")

    # ------------------------------------------------------------------ inference
    def embed(self, clips_uint8: np.ndarray) -> np.ndarray:
        """uint8 [b, 13, 182, 182, 3] -> float32 X3D-S features [b, 2048]."""
        with self.lock, torch.inference_mode():
            x = torch.from_numpy(clips_uint8).to(self.device_name)
            x = x.permute(0, 4, 1, 2, 3).to(self.x3d_dtype).div_(255).sub_(self._mean).div_(self._std)
            return self.x3d(x).float().cpu().numpy()

    def mil_raw(self, feats: np.ndarray) -> np.ndarray:
        """Raw (unsmoothed) Stage 1 scores for float [n, 2048] features."""
        with self.lock, torch.inference_mode():
            x = torch.from_numpy(l2_normalise(feats)).to(self.device_name)
            return self.mil(x).float().cpu().numpy().ravel()

    def clip_window(self, frames: list[Image.Image], timestamps: list[float], suppress: float,
                    details: bool = True) -> dict[str, Any]:
        with self.lock:
            return self.clip.score_window(frames, frame_timestamps=timestamps, suppress_threshold=suppress, top_k=3,
                                          details=details)

    def mil_trace(self, feat: np.ndarray) -> dict[str, Any]:
        """Stage 1 head internals for one X3D feature [2048]: L2-normalised input -> fc1 (ReLU) -> fc2 -> sigmoid."""
        m = self.mil
        with self.lock, torch.inference_mode():
            x = torch.from_numpy(l2_normalise(feat[None].astype(np.float32))).to(self.device_name)
            h1 = torch.relu(m.fc1(x))
            h2 = m.fc2(h1)
            logit = m.fc3(h2)
            return {"hidden1": h1[0].float().cpu().numpy(), "hidden2": h2[0].float().cpu().numpy(),
                    "logit": float(logit.item()), "score": float(torch.sigmoid(logit).item())}


# ====================================================================== uploads
class Job:
    def __init__(self, job_id: str, name: str, path: Path, tau: float, suppress: float, tau_a: float = DEFAULT_AUDIO_TAU) -> None:
        self.id, self.name, self.path = job_id, name, path
        self.dir = path.parent
        self.tau, self.suppress, self.tau_a = tau, suppress, tau_a
        self.wav32: np.ndarray | None = None
        self.wav44: np.ndarray | None = None
        self.audio_trig = None
        self.status, self.progress, self.message, self.error = "queued", 0.0, "Waiting", ""
        self.result: dict[str, Any] | None = None
        self.generation = 0
        self.created = time.time()
        # cached per-video Stage 1 data (re-used when τ changes)
        self.raw: np.ndarray | None = None
        self.smoothed: np.ndarray | None = None
        self.feats: np.ndarray | None = None   # [n_clips, 2048] X3D-S features, for the stage inspector
        self.n_frames = 0
        self.fps = 30.0
        self.timing: dict[str, float] = {}

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "status": self.status, "progress": round(self.progress, 4),
            "message": self.message, "error": self.error, "tau": self.tau, "suppress": self.suppress,
            "tau_a": self.tau_a, "result": self.result,
        }


class JobManager:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.jobs: dict[str, Job] = {}
        self.work: queue.Queue[tuple[str, Job]] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name="upload-worker").start()

    def create(self, filename: str, fileobj, tau: float | None, suppress: float | None,
               tau_a: float | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        jdir = DATA_DIR / "jobs" / job_id
        jdir.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix.lower() or ".mp4"
        path = jdir / f"source{ext}"
        with open(path, "wb") as f:
            shutil.copyfileobj(fileobj, f, length=4 * 1024 * 1024)
        job = Job(job_id, Path(filename).name, path,
                  tau if tau is not None else (self.engine.tau or 0.5),
                  suppress if suppress is not None else DEFAULT_SUPPRESS_THRESHOLD,
                  tau_a if tau_a is not None else self.engine.tau_a)
        self.jobs[job_id] = job
        self.work.put(("analyze", job))
        return job

    def rescore(self, job: Job, tau: float, suppress: float, tau_a: float | None = None) -> None:
        if job.raw is None:
            raise RuntimeError("video has not been analysed yet")
        job.tau, job.suppress = float(tau), float(suppress)
        if tau_a is not None:
            job.tau_a = float(tau_a)
        job.status, job.progress, job.message = "queued", 0.0, "Re-scoring with new thresholds"
        self.work.put(("classify", job))

    def _worker(self) -> None:
        while True:
            kind, job = self.work.get()
            try:
                self.engine.require_ready()
                if kind == "analyze":
                    self._extract(job)
                self._classify(job)
                job.status, job.progress, job.message = "done", 1.0, "Done"
            except Exception as exc:
                traceback.print_exc()
                job.status, job.error, job.message = "error", f"{type(exc).__name__}: {exc}", "Failed"

    def _extract(self, job: Job) -> None:
        job.status, job.message = "extracting", "Stage 1: X3D-S features"
        cap = cv2.VideoCapture(str(job.path))
        if not cap.isOpened():
            raise IOError("OpenCV cannot open this video (try an .mp4 with H.264)")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        t0 = time.time()
        feats: list[np.ndarray] = []
        gen = iter_clip_batches(str(job.path), batch=8)
        t_model = 0.0
        while True:
            try:
                arr = next(gen)
            except StopIteration as stop:
                n_raw, fps = stop.value
                break
            tm = time.time()
            feats.append(self.engine.embed(arr))
            t_model += time.time() - tm
            done = sum(len(f) for f in feats) * FRAMES_PER_CLIP
            job.progress = min(0.9, 0.9 * done / total) if total else 0.0
            job.message = f"Stage 1: X3D-S features ({sum(len(f) for f in feats)} clips)"
        if not feats:
            raise IOError("no frames decoded")
        f = np.concatenate(feats)
        tm = time.time()
        job.feats = f.astype(np.float16)
        job.raw = self.engine.mil_raw(f)
        t_head = time.time() - tm
        job.smoothed = smooth_scores(job.raw, window=self.engine.smooth_window)
        job.n_frames, job.fps = int(n_raw), float(fps or 30.0)
        n = len(f)
        job.timing = {
            "extract_total_sec": round(time.time() - t0, 3),
            "x3d_ms_per_clip": round(1000 * t_model / n, 2),
            "decode_ms_per_clip": round(1000 * (time.time() - t0 - t_model) / n, 2),
            "mil_head_ms_total": round(1000 * t_head, 2),
        }
        # audio Stage 1 once per upload (re-gated on rescore)
        if self.engine.audio_state == "ready":
            job.message = "Audio Stage 1: EfficientAT trigger"
            ta = time.time()
            job.wav32 = load_audio(job.path, EAT_SR)
            if job.wav32 is not None and len(job.wav32):
                job.wav44 = load_audio(job.path, CLAP_SR)
                job.audio_trig = self.engine.audio_score(job.wav32)
            job.timing["audio_trigger_sec"] = round(time.time() - ta, 3)

    def _classify(self, job: Job) -> None:
        job.status, job.progress, job.message = "classifying", 0.92, "Stage 2: CLIP on flagged windows"
        job.generation += 1
        gen = job.generation
        peaks = job.dir / "peaks"
        if peaks.exists():
            shutil.rmtree(peaks)
        peaks.mkdir(parents=True, exist_ok=True)
        assert job.smoothed is not None and job.raw is not None
        wins = extract_flagged_windows(job.smoothed, tau=job.tau, n_frames=job.n_frames, fps=job.fps,
                                       frames_per_clip=FRAMES_PER_CLIP)
        events, suppressed = [], []
        t0 = time.time()
        for k, w in enumerate(wins):
            frames, idxs, ts = sample_window_frames(job.path, w.start_frame, w.end_frame, num_frames=8, fps=job.fps)
            res = self.engine.clip_window(frames, ts, job.suppress)
            p = int(res["peak_frame_idx"])
            fname = f"g{gen}_w{k:02d}.jpg"
            frames[p].save(peaks / fname, quality=88)
            if "details" in res:  # every frame CLIP looked at, as small thumbnails for the inspector
                urls = []
                for i, fr in enumerate(frames):
                    tn = fr.convert("RGB")
                    tn.thumbnail((240, 240))
                    tn.save(peaks / f"g{gen}_w{k:02d}_f{i}.jpg", quality=80)
                    urls.append(f"/api/jobs/{job.id}/peaks/g{gen}_w{k:02d}_f{i}.jpg")
                res["details"].update(frame_ts=[round(float(t), 2) for t in ts], frame_urls=urls,
                                      frame_idx=[int(i) for i in idxs], peak_idx=p)
            ev = {
                "id": k,
                "start_sec": w.start_sec, "end_sec": w.end_sec,
                "start_clip": w.start_clip, "end_clip": w.end_clip,
                "stage1_peak": round(w.peak_score, 4),
                "peak_ts": ts[p],
                "peak_url": f"/api/jobs/{job.id}/peaks/{fname}",
                **_event_fields(res),
            }
            (suppressed if ev["suppressed"] else events).append(ev)
            job.progress = 0.92 + 0.08 * (k + 1) / max(1, len(wins))
        audio = self._audio_branch(job)
        fused = fuse(events, suppressed, audio["events"]) if audio else []
        clip_sec = FRAMES_PER_CLIP / job.fps
        n = len(job.smoothed)
        flagged = int(np.sum(job.smoothed > job.tau))
        job.timing["clip_stage_sec"] = round(time.time() - t0, 3)
        job.result = {
            "name": job.name,
            "fps": round(job.fps, 3),
            "n_frames": job.n_frames,
            "duration_sec": round(job.n_frames / job.fps, 2),
            "n_clips": n,
            "clip_sec": round(clip_sec, 4),
            "tau": job.tau,
            "suppress": job.suppress,
            "scores": [round(float(s), 4) for s in job.smoothed],
            "raw_scores": [round(float(s), 4) for s in job.raw],
            "flagged_clips": flagged,
            "escalated_pct": round(100.0 * flagged / max(1, n), 2),
            "windows": [{"start_sec": w.start_sec, "end_sec": w.end_sec, "start_clip": w.start_clip,
                         "end_clip": w.end_clip, "raw_start": w.raw_start, "raw_end": w.raw_end,
                         "peak_score": round(w.peak_score, 4)} for w in wins],
            "events": events,
            "suppressed_events": suppressed,
            "timing": job.timing,
            "device": self.engine.device_name,
            "audio": audio,
            "fused_alerts": fused,
        }

    def _audio_branch(self, job: Job) -> dict[str, Any] | None:
        """Audio Stage 1 gate + Stage 2 CLAP for an uploaded video (None if audio models aren't loaded)."""
        if self.engine.audio_state != "ready":
            return None
        trig = job.audio_trig
        if trig is None:
            return {"has_audio": False, "silent_frac": 1.0, "scores": [], "times": [], "windows": [], "events": [],
                    "suppressed_events": [], "tau_a": job.tau_a}
        job.message = "Audio Stage 2: CLAP on flagged audio"
        wins = audio_windows(trig, job.tau_a)
        events, suppressed = [], []
        t0 = time.time()
        for k, w in enumerate(wins):
            seg = job.wav44[int(w["start_sec"] * CLAP_SR): int(w["end_sec"] * CLAP_SR)]
            res = self.engine.clap_segment(seg, CLAP_SR, job.suppress)
            ev = {"id": k, **w, **res, "video_class": AUDIO_TO_VIDEO_CLASS.get(res["class"])}
            (suppressed if res["suppressed"] else events).append(ev)
        job.timing["clap_stage_sec"] = round(time.time() - t0, 3)
        return {
            "has_audio": True,
            "silent_frac": round(float(trig.silent.mean()), 4),
            "times": [round(float(t), 2) for t in trig.times],
            "scores": [round(float(s), 4) for s in trig.scores],
            "top_group": trig.top_group,
            "tau_a": job.tau_a,
            "windows": [{"start_sec": w["start_sec"], "end_sec": w["end_sec"]} for w in wins],
            "events": events,
            "suppressed_events": suppressed,
            "group_scores": {g: [round(float(x), 4) for x in v] for g, v in trig.group_scores.items()},
            "silent": [bool(x) for x in trig.silent],
            "dbfs": [round(float(x), 1) for x in trig.dbfs],
            "model": getattr(self.engine.trigger, "mode", None),
        }

    # ------------------------------------------------------------------ stage inspector
    def clip_detail(self, job: Job, k: int) -> dict[str, Any]:
        """Everything Stage 1 computed for one 2.6 s clip: X3D-S embedding, MIL head internals, raw/smoothed score."""
        if job.feats is None or job.raw is None or job.smoothed is None:
            raise RuntimeError("video has not been analysed yet")
        if not 0 <= k < len(job.raw):
            raise IndexError("clip index out of range")
        f = job.feats[k].astype(np.float32)
        tr = self.engine.mil_trace(f)
        w = self.engine.smooth_window
        lo, hi = max(0, k - (w - 1) // 2), min(len(job.raw), k + w // 2 + 1)
        first = k * FRAMES_PER_CLIP
        thumb = None
        cap = cv2.VideoCapture(str(job.path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first + FRAMES_PER_CLIP // 2)
            ok, bgr = cap.read()
            if ok:
                thumb = _jpeg_data_url(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), max_w=360)
        finally:
            cap.release()
        return {
            "clip": k, "t0": round(first / job.fps, 2), "t1": round((first + FRAMES_PER_CLIP) / job.fps, 2),
            "frames": [first, first + FRAMES_PER_CLIP - 1], "frames_sampled": CLIP_LEN, "frame_stride": STRIDE,
            "raw": round(float(job.raw[k]), 4), "smoothed": round(float(job.smoothed[k]), 4),
            "smooth_from": [lo, hi - 1], "smooth_window": w, "tau": job.tau,
            "above_tau": bool(job.smoothed[k] > job.tau),
            "embedding": [round(float(x), 4) for x in f], "embedding_model": "X3D-S (Kinetics-400), frozen",
            "embedding_norm": round(float(np.linalg.norm(f)), 3),
            "hidden1": [round(float(x), 4) for x in tr["hidden1"]], "hidden2": [round(float(x), 4) for x in tr["hidden2"]],
            "logit": round(tr["logit"], 4), "head_score": round(tr["score"], 4),
            "thumb": thumb,
        }

    def audio_detail(self, job: Job, k: int) -> dict[str, Any]:
        """Everything audio Stage 1 computed for one 1 s window: EfficientAT embedding and per-class scores."""
        trig = job.audio_trig
        if trig is None:
            raise RuntimeError("no audio for this video")
        if not 0 <= k < len(trig.scores):
            raise IndexError("audio window out of range")
        feats = trig.feats[k] if trig.feats is not None else None
        return {
            "window": k, "t0": float(trig.times[k]), "t1": float(trig.times[k]) + 1.0,
            "dbfs": round(float(trig.dbfs[k]), 1), "silent": bool(trig.silent[k]),
            "score": round(float(trig.scores[k]), 4), "tau_a": job.tau_a, "above_tau": bool(trig.scores[k] > job.tau_a),
            "top_group": trig.top_group[k],
            "group_scores": {g: round(float(v[k]), 4) for g, v in trig.group_scores.items()},
            "model": getattr(self.engine.trigger, "mode", None),
            "embedding": [round(float(x), 4) for x in feats] if feats is not None else [],
            "embedding_model": "EfficientAT mn10 (AudioSet), frozen; 1 s tiled to 10 s",
        }


# ====================================================================== live camera
class LiveSession:
    """Streams webcam frames through the same cascade, clip by clip.

    The browser sends JPEG frames at 5 fps (= every 6th frame of 30 fps video, the
    X3D-S sampling rate), so 13 frames make one 2.6 s clip. Scores use the same
    centred 3-clip moving average as offline, which means each clip's smoothed
    score is final one clip (2.6 s) after it arrives. Windows follow §3.4: clips
    > τ, merged across gaps of < 2 clips, padded ±1 clip; CLIP re-scores the
    window on each new flagged clip ("active") and once more when it closes ("final").
    """

    FPS = 30.0 / STRIDE        # 5 frames per second
    MAX_BACKLOG = 4            # clips waiting for the model before we drop
    MAX_WINDOW_CLIPS = 46      # ~2 min: force-close very long windows

    AUDIO_MAGIC = b"AUD0"      # binary messages starting with this are 1 s of int16 PCM at 32 kHz

    def __init__(self, engine: Engine, emit: Callable[[dict[str, Any]], None], tau: float, suppress: float,
                 tau_a: float | None = None) -> None:
        self.engine, self.emit = engine, emit
        self.tau, self.suppress = float(tau), float(suppress)
        self.tau_a = float(engine.tau_a if tau_a is None else tau_a)
        self.achunks: dict[int, np.ndarray] = {}
        self.a_n = 0
        self.awin: dict[str, Any] | None = None
        self.n_aevents = 0
        self.v_events: dict[int, dict] = {}
        self.a_events: dict[int, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self.pending: list[bytes] = []
        self.frames_in = 0
        self.raw: list[float] = []
        self.clips: dict[int, tuple[int, list[bytes]]] = {}  # clip idx -> (first frame no., jpegs)
        self.finalized = -1
        self.win: dict[str, Any] | None = None
        self.n_events = 0
        self.dropped = 0
        self.closed = False
        w = engine.smooth_window
        self.half_left, self.half_right = (w - 1) // 2, w - 1 - (w - 1) // 2
        self.thread = threading.Thread(target=self._run, daemon=True, name="live-session")
        self.thread.start()

    # -------------------------------------------------------------- input side
    def push_frame(self, jpeg: bytes) -> None:
        if self.closed:
            return
        self.pending.append(jpeg)
        self.frames_in += 1
        if len(self.pending) == CLIP_LEN:
            first = self.frames_in - CLIP_LEN
            if self.q.qsize() >= self.MAX_BACKLOG:
                self.dropped += 1
                self.emit({"type": "warning", "message": f"Model is falling behind; dropped a clip ({self.dropped} so far)"})
            else:
                self.q.put(("clip", first, self.pending))
            self.pending = []

    def push_audio(self, payload: bytes) -> None:
        if self.closed or self.engine.audio_state != "ready":
            return
        if self.q.qsize() >= self.MAX_BACKLOG * 3:
            return  # drop audio rather than fall further behind
        self.q.put(("audio", 0, payload[len(self.AUDIO_MAGIC):]))

    def set_params(self, tau: float | None = None, suppress: float | None = None, tau_a: float | None = None) -> None:
        if tau is not None:
            self.tau = float(tau)
        if suppress is not None:
            self.suppress = float(suppress)
        if tau_a is not None:
            self.tau_a = float(tau_a)

    def flush(self) -> None:
        self.q.put(("flush", 0, []))

    def close(self) -> None:
        self.closed = True
        self.q.put(("quit", 0, []))

    # -------------------------------------------------------------- worker side
    def _run(self) -> None:
        while True:
            kind, first, jpegs = self.q.get()
            try:
                if kind == "quit":
                    return
                if kind == "flush":
                    self._flush()
                    self.emit({"type": "stopped"})
                    continue
                if kind == "audio":
                    self._process_audio(jpegs)  # type: ignore[arg-type]
                    continue
                self._process_clip(first, jpegs)
            except Exception as exc:
                traceback.print_exc()
                self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _process_clip(self, first: int, jpegs: list[bytes]) -> None:
        bgr = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in jpegs]
        arr = np.stack([_prep(im) for im in bgr])[None]
        t0 = time.perf_counter()
        feat = self.engine.embed(arr)
        t1 = time.perf_counter()
        raw = float(self.engine.mil_raw(feat)[0])
        t2 = time.perf_counter()
        j = len(self.raw)
        self.raw.append(raw)
        self.clips[j] = (first, jpegs)
        self.emit({"type": "timing", "x3d_ms": round(1000 * (t1 - t0), 1), "mil_ms": round(1000 * (t2 - t1), 2),
                   "backlog": self.q.qsize()})
        # a clip's centred average is final once `half_right` later clips exist
        while self.finalized + 1 + self.half_right <= j:
            self._finalize(self.finalized + 1)

    def _flush(self) -> None:
        while self.finalized + 1 < len(self.raw):
            self._finalize(self.finalized + 1)
        if self.win is not None:
            self._close_window()
        if self.awin is not None:
            self._close_awin()

    # -------------------------------------------------------------- audio (1 s chunks)
    def _process_audio(self, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if x.size == 0:
            return
        k = self.a_n
        self.a_n += 1
        self.achunks[k] = x
        t0 = time.perf_counter()
        tr = self.engine.audio_score(x)
        s = float(tr.scores[0])
        self.emit({"type": "ascore", "idx": k, "t0": float(k), "t1": float(k + 1), "score": round(s, 4),
                   "group": tr.top_group[0], "silent": bool(tr.silent[0]), "dbfs": round(float(tr.dbfs[0]), 1),
                   "flagged": s > self.tau_a, "ms": round(1000 * (time.perf_counter() - t0), 1)})
        if s > self.tau_a:
            if self.awin is None:
                self.awin = {"id": self.n_aevents, "start": max(0, k - 1), "last": k, "group": tr.top_group[0]}
                self.n_aevents += 1
            else:
                self.awin["last"] = k
            self._classify_audio(self.awin["start"], k, "active")
        elif self.awin is not None and k - self.awin["last"] - 1 >= 2:
            self._close_awin()
        keep = self.awin["start"] if self.awin is not None else k - 1
        for i in [i for i in self.achunks if i < keep]:
            del self.achunks[i]

    def _close_awin(self) -> None:
        assert self.awin is not None
        self._classify_audio(self.awin["start"], min(self.awin["last"] + 1, self.a_n - 1), "final")
        self.awin = None

    def _classify_audio(self, a: int, b: int, status: str) -> None:
        w = self.awin
        parts = [self.achunks[i] for i in range(a, b + 1) if i in self.achunks]
        if not parts or w is None:
            return
        t0 = time.perf_counter()
        res = self.engine.clap_segment(np.concatenate(parts), EAT_SR, self.suppress)
        ev = {"type": "aevent", "id": w["id"], "status": status, "start_sec": float(a), "end_sec": float(b + 1),
              "trigger_group": w["group"], "video_class": AUDIO_TO_VIDEO_CLASS.get(res["class"]),
              "clap_ms": round(1000 * (time.perf_counter() - t0), 1), **res}
        self.emit(ev)
        self.a_events[w["id"]] = ev
        self._emit_fused()

    def _emit_fused(self) -> None:
        v = [e for e in self.v_events.values() if not e["suppressed"]]
        vs = [e for e in self.v_events.values() if e["suppressed"]]
        a = [e for e in self.a_events.values() if not e["suppressed"]]
        self.emit({"type": "fused", "alerts": fuse(v, vs, a)})

    def _clip_t(self, idx: int) -> tuple[float, float]:
        first = self.clips[idx][0] if idx in self.clips else idx * CLIP_LEN
        return round(first / self.FPS, 2), round((first + CLIP_LEN) / self.FPS, 2)

    def _finalize(self, k: int) -> None:
        lo, hi = max(0, k - self.half_left), min(len(self.raw), k + self.half_right + 1)
        s = float(np.mean(self.raw[lo:hi]))
        self.finalized = k
        flagged = s > self.tau
        t0, t1 = self._clip_t(k)
        self.emit({"type": "score", "clip": k, "t0": t0, "t1": t1, "score": round(s, 4),
                   "raw": round(self.raw[k], 4), "flagged": flagged, "tau": self.tau})
        last_clip = len(self.raw) - 1
        if flagged:
            if self.win is None:
                self.win = {"id": self.n_events, "start": max(0, k - 1), "last": k, "res": None, "range": None}
                self.n_events += 1
            else:
                self.win["last"] = k
            self._classify(self.win["start"], min(k + 1, last_clip), status="active")
            if self.win["last"] - self.win["start"] + 1 >= self.MAX_WINDOW_CLIPS:
                self._close_window()
        elif self.win is not None and k - self.win["last"] - 1 >= 2:
            self._close_window()
        # free JPEGs no window can need any more
        keep_from = self.win["start"] if self.win is not None else k - 1
        for idx in [i for i in self.clips if i < keep_from]:
            del self.clips[idx]

    def _close_window(self) -> None:
        assert self.win is not None
        end = min(self.win["last"] + 1, len(self.raw) - 1)
        self._classify(self.win["start"], end, status="final")
        self.win = None

    def _classify(self, a: int, b: int, status: str) -> None:
        win = self.win
        assert win is not None
        if win["range"] == (a, b) and win["res"] is not None:
            res, peak_img, peak_ts = win["res"]
        else:
            frames: list[tuple[int, bytes]] = []
            for idx in range(a, b + 1):
                if idx in self.clips:
                    first, jp = self.clips[idx]
                    frames.extend((first + i, x) for i, x in enumerate(jp))
            if not frames:
                return
            n = len(frames)
            pick = [frames[min(n - 1, int(n * (i + 0.5) / 8))] for i in range(8)]
            imgs = [Image.open(io.BytesIO(x)).convert("RGB") for _, x in pick]
            ts = [round(f / self.FPS, 2) for f, _ in pick]
            t0 = time.perf_counter()
            res = self.engine.clip_window(imgs, ts, self.suppress)
            self.emit({"type": "timing", "clip_ms": round(1000 * (time.perf_counter() - t0), 1)})
            p = int(res["peak_frame_idx"])
            peak_img, peak_ts = imgs[p], ts[p]
            win["range"], win["res"] = (a, b), (res, peak_img, peak_ts)
        start, _ = self._clip_t(a)
        _, end = self._clip_t(b)
        ev = {
            "type": "event", "id": win["id"], "status": status,
            "start_sec": start, "end_sec": end, "n_clips": b - a + 1,
            "peak_ts": peak_ts, "peak_jpeg": _jpeg_data_url(peak_img),
            **_event_fields(res),
        }
        self.emit(ev)
        self.v_events[win["id"]] = {k: v for k, v in ev.items() if k != "peak_jpeg"}
        if self.a_events:
            self._emit_fused()
