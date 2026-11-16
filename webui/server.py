"""FastAPI server for the anomaly-detection web UI.

Run from the project folder:  ./run_webui.sh   (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webui.engine import DATA_DIR, ROOT, Engine, JobManager, LiveSession

STATIC = ROOT / "webui" / "static"
engine = Engine()
jobs = JobManager(engine)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=engine.load, daemon=True, name="model-loader").start()
    yield


app = FastAPI(title="Video Anomaly Detection", lifespan=lifespan)


@app.get("/api/status")
def status() -> dict:
    return engine.status()


@app.post("/api/reload")
def reload_models() -> dict:
    if engine.state != "loading":
        engine.state = "idle"
        threading.Thread(target=engine.load, daemon=True).start()
    return {"ok": True}


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...), tau: float | None = Form(None), suppress: float | None = Form(None),
               tau_a: float | None = Form(None)) -> dict:
    if engine.state != "ready":
        raise HTTPException(503, engine.error or f"Models are {engine.state}")
    job = jobs.create(file.filename or "video.mp4", file.file, tau, suppress, tau_a)
    return job.public()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.public()


class Rescore(BaseModel):
    tau: float
    suppress: float
    tau_a: float | None = None


@app.post("/api/jobs/{job_id}/rescore")
def rescore(job_id: str, body: Rescore) -> dict:
    job = jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.status in ("queued", "extracting", "classifying"):
        raise HTTPException(409, "job is still running")
    jobs.rescore(job, body.tau, body.suppress, body.tau_a)
    return job.public()


@app.get("/api/jobs/{job_id}/peaks/{fname}")
def peak(job_id: str, fname: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id) or not re.fullmatch(r"g\d+_w\d+(_f\d+)?\.jpg", fname):
        raise HTTPException(404)
    p = DATA_DIR / "jobs" / job_id / "peaks" / fname
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


def _inspect(job_id: str, fn, k: int) -> dict:
    job = jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    try:
        return fn(job, k)
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/jobs/{job_id}/clips/{k}")
def clip_detail(job_id: str, k: int) -> dict:
    """Stage 1 output for one 2.6 s clip (X3D-S embedding + MIL head internals)."""
    return _inspect(job_id, jobs.clip_detail, k)


@app.get("/api/jobs/{job_id}/audio/{k}")
def audio_detail(job_id: str, k: int) -> dict:
    """Audio Stage 1 output for one 1 s window (EfficientAT embedding + per-class scores)."""
    return _inspect(job_id, jobs.audio_detail, k)


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    if engine.state != "ready":
        await ws.send_json({"type": "error", "message": engine.error or f"Models are {engine.state}"})
        await ws.close()
        return
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue()

    def emit(msg: dict) -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(out.put_nowait, msg)

    async def sender() -> None:
        while True:
            msg = await out.get()
            await ws.send_json(msg)

    session: LiveSession | None = None
    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                if session is not None:
                    b = msg["bytes"]
                    if b[:4] == LiveSession.AUDIO_MAGIC:
                        session.push_audio(b)
                    else:
                        session.push_frame(b)
                continue
            data = json.loads(msg.get("text") or "{}")
            kind = data.get("type")
            if kind == "start":
                if session is not None:
                    session.close()
                t, s, ta = data.get("tau"), data.get("suppress"), data.get("tau_a")
                session = LiveSession(engine, emit, tau=float(engine.tau if t is None else t),
                                      suppress=float(0.6 if s is None else s), tau_a=ta)
                emit({"type": "ready", "tau": session.tau, "suppress": session.suppress, "tau_a": session.tau_a,
                      "device": engine.device_name, "fps": LiveSession.FPS, "audio": engine.audio_state == "ready",
                      "audio_sr": 32000})
            elif kind == "params" and session is not None:
                session.set_params(data.get("tau"), data.get("suppress"), data.get("tau_a"))
            elif kind == "stop" and session is not None:
                session.flush()
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            session.close()
        await asyncio.sleep(0.05)
        send_task.cancel()


@app.exception_handler(RuntimeError)
async def runtime_error(_req, exc: RuntimeError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
