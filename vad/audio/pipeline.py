"""Two-stage audio branch + decision-level audio/video fusion (research report, recommendation P2).

Audio Stage 1  EfficientAT mn10 AudioSet trigger on 1 s windows (score = max prob over trigger classes;
               digitally silent windows score 0).
Gate           windows with score > τ_a; merge windows < 2 apart; pad ±1 window (same rule as video §3.4).
Audio Stage 2  MS-CLAP 2023 zero-shot over 7 classes; suppress if P(normal)+P(unrelated) >= 0.6.
Fusion         per-class weighted average of video (CLIP) and audio (CLAP) class probabilities where a
               video window and an audio event overlap; audio weight is 0 when audio is silent or not
               flagged. A CLIP-suppressed video window is promoted only by a confident audio shooting /
               explosion event; audio events with no video window become audio-only (off-camera) alerts.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vad.audio.audio_io import load_audio
from vad.audio.clap_scorer import AUDIO_TO_VIDEO_CLASS, CLAP_SR, DEFAULT_SUPPRESS_THRESHOLD, ClapScorer
from vad.audio.trigger import DEFAULT_AUDIO_TAU, EAT_SR, WINDOW_SEC, AudioTrigger, TriggerResult
from vad.pipeline.run_video import extract_flagged_windows

# per-class audio weight in fusion (prior from the literature: audio helps shooting/explosion most,
# fighting somewhat, burglary/fire little; HL-Net per-class ablation). Not tuned on our test data.
AUDIO_FUSION_WEIGHT: dict[str, float] = {"shooting": 0.5, "explosion": 0.5, "fighting": 0.3, "burglary": 0.2, "fire": 0.2}
PROMOTE_CLASSES = ("shooting", "explosion")
PROMOTE_MIN_CONF = 0.5


def audio_windows(trig: TriggerResult, tau_a: float) -> list[dict[str, Any]]:
    """Gate 1 s trigger windows with the same merge/pad rule as the video windows."""
    W = len(trig.scores)
    wins = extract_flagged_windows(trig.scores, tau=tau_a, n_frames=W, fps=1.0 / WINDOW_SEC, frames_per_clip=1)
    out = []
    for w in wins:
        seg = slice(w.raw_start, w.raw_end + 1)
        g = {k: float(v[seg].max()) for k, v in trig.group_scores.items()}
        out.append({"start_sec": round(w.start_clip * WINDOW_SEC, 2), "end_sec": round((w.end_clip + 1) * WINDOW_SEC, 2),
                    "trigger_peak": round(w.peak_score, 4), "trigger_group": max(g, key=g.get),
                    "trigger_groups": {k: round(v, 4) for k, v in g.items()}})
    return out


def run_audio(
    source: str | np.ndarray,
    trigger: AudioTrigger,
    clap: ClapScorer | None,
    *,
    tau_a: float = DEFAULT_AUDIO_TAU,
    suppress: float = DEFAULT_SUPPRESS_THRESHOLD,
    wav44k: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run audio Stage 1 (+ Stage 2 if `clap`) on a file path or a 32 kHz waveform."""
    wav32 = load_audio(source, EAT_SR) if isinstance(source, str) else source
    if wav32 is None or len(wav32) == 0:
        return {"has_audio": False, "silent_frac": 1.0, "times": [], "scores": [], "events": [], "suppressed_events": [],
                "windows": []}
    trig = trigger.score(wav32)
    wins = audio_windows(trig, tau_a)
    events, suppressed = [], []
    if clap is not None and wins:
        if wav44k is None:
            wav44k = load_audio(source, CLAP_SR) if isinstance(source, str) else None
        for k, w in enumerate(wins):
            if wav44k is not None:
                seg = wav44k[int(w["start_sec"] * CLAP_SR): int(w["end_sec"] * CLAP_SR)]
                res = clap.score_segment(seg, CLAP_SR, suppress_threshold=suppress)
            else:
                seg = wav32[int(w["start_sec"] * EAT_SR): int(w["end_sec"] * EAT_SR)]
                res = clap.score_segment(seg, EAT_SR, suppress_threshold=suppress)
            ev = {"id": k, **w, **res, "video_class": AUDIO_TO_VIDEO_CLASS.get(res["class"])}
            (suppressed if res["suppressed"] else events).append(ev)
    return {
        "has_audio": True,
        "duration_sec": round(len(wav32) / EAT_SR, 2),
        "silent_frac": round(float(trig.silent.mean()), 4),
        "times": trig.times.tolist(),
        "scores": [round(float(s), 4) for s in trig.scores],
        "top_group": trig.top_group,
        "tau_a": tau_a,
        "windows": wins,
        "events": events,
        "suppressed_events": suppressed,
        "_trigger": trig,
    }


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) > max(a0, b0)


AUDIO_TO_VIDEO_ALL = {**AUDIO_TO_VIDEO_CLASS, "normal": "normal", "unrelated": "unrelated"}
NON_ANOMALY_WEIGHT = 0.3  # audio weight for normal/unrelated so a benign soundtrack can pull a window back to benign


def _audio_probs_in_video_space(audio_ev: dict[str, Any]) -> dict[str, float]:
    return {AUDIO_TO_VIDEO_ALL[c]: p for c, p in audio_ev["mean_probs"].items() if c in AUDIO_TO_VIDEO_ALL}


def _w(c: str) -> float:
    return NON_ANOMALY_WEIGHT if c in ("normal", "unrelated") else AUDIO_FUSION_WEIGHT.get(c, 0.0)


def fuse(video_events: list[dict], video_suppressed: list[dict], audio_events: list[dict]) -> list[dict[str, Any]]:
    """Decision-level fusion. Video events must carry `mean_probs` (9 CLIP classes)."""
    alerts: list[dict[str, Any]] = []
    used_audio: set[int] = set()
    for ev, was_suppressed in [(e, False) for e in video_events] + [(e, True) for e in video_suppressed]:
        ov = [a for a in audio_events if _overlap(ev["start_sec"], ev["end_sec"], a["start_sec"], a["end_sec"])]
        used_audio.update(a["id"] for a in ov)
        if not ov:
            if not was_suppressed:
                alerts.append({**_alert_base(ev), "class": ev["class"], "confidence": ev["confidence"], "source": "video"})
            continue
        best_a = max(ov, key=lambda a: a["confidence"])
        pa = _audio_probs_in_video_space(best_a)
        pv = ev.get("mean_probs") or {t["class"]: t["prob"] for t in ev["top3"]}
        # fuse over ALL classes (incl. normal/unrelated): an argmax over anomaly classes only would invent an
        # anomaly class when both modalities say "unrelated" (seen 20 Nov: unrelated+unrelated -> 'shooting 18%')
        fused = {c: (1 - _w(c)) * pv.get(c, 0.0) + _w(c) * pa.get(c, 0.0) for c in pv}
        top = max(fused, key=fused.get)
        confident_audio = bool(best_a.get("video_class")) and best_a["confidence"] >= PROMOTE_MIN_CONF
        if top in ("normal", "unrelated") and confident_audio and not (was_suppressed and best_a["video_class"] in PROMOTE_CLASSES):
            # (22 Nov) the window looks benign to CLIP (or was vetoed) but the audio heard a confident anomaly, e.g.
            # glass breaking off-camera: raise the audio alert instead of letting the benign verdict swallow it.
            # Vetoed windows with a confident gunshot/explosion keep the original 'audio-confirmed' path below.
            alerts.append({"start_sec": best_a["start_sec"], "end_sec": best_a["end_sec"], "class": best_a["video_class"],
                           "confidence": best_a["confidence"], "source": "audio", "audio_class": best_a["class"],
                           "video_class": ev["class"]})
            continue
        if not was_suppressed:
            if top in ("normal", "unrelated"):  # not vetoed => still an alert; name its most likely anomaly class
                top = max((c for c in fused if c not in ("normal", "unrelated")), key=fused.get, default=top)
            alerts.append({**_alert_base(ev), "class": top, "confidence": round(fused[top], 4), "source": "audio+video",
                           "video_class": ev["class"], "audio_class": best_a["class"], "fused_probs": _r(fused)})
        elif best_a.get("video_class") in PROMOTE_CLASSES and best_a["confidence"] >= PROMOTE_MIN_CONF:
            alerts.append({**_alert_base(ev), "class": best_a["video_class"], "confidence": best_a["confidence"],
                           "source": "audio-confirmed", "video_class": ev["class"], "audio_class": best_a["class"]})
    for a in audio_events:
        if a["id"] not in used_audio and a.get("video_class"):
            alerts.append({"start_sec": a["start_sec"], "end_sec": a["end_sec"], "class": a["video_class"],
                           "confidence": a["confidence"], "source": "audio", "audio_class": a["class"]})
    return sorted(alerts, key=lambda x: x["start_sec"])


def _alert_base(ev: dict) -> dict:
    return {"start_sec": ev["start_sec"], "end_sec": ev["end_sec"]}


def _r(d: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in d.items()}
