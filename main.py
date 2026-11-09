"""
FastAPI backend for the intelligent video surveillance dashboard.

Endpoints:
    GET  /                      -> dashboard UI
    POST /api/upload            -> upload a video file, get anomaly timeline
    WS   /ws/webcam              -> stream webcam frames (base64 JPEG) from
                                    browser, get live anomaly scores back
    POST /api/rtsp/start        -> start monitoring an RTSP stream
    POST /api/rtsp/stop         -> stop the RTSP monitor
    GET  /api/rtsp/status       -> poll latest RTSP anomaly result
    GET  /api/health            -> model/threshold status
"""

import base64
import os
import shutil
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import AnomalyDetector

MODEL_PATH = os.environ.get("MODEL_PATH", "../models/autoencoder.keras")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Intelligent Video Surveillance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: Optional[AnomalyDetector] = None
_detector_error: Optional[str] = None


def get_detector() -> AnomalyDetector:
    global _detector, _detector_error
    if _detector is None and _detector_error is None:
        try:
            _detector = AnomalyDetector(MODEL_PATH)
        except Exception as e:  # noqa: BLE001
            _detector_error = str(e)
    if _detector is None:
        raise RuntimeError(_detector_error)
    return _detector


# ------------------------------------------------------------------
# Static dashboard
# ------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/health")
def health():
    try:
        det = get_detector()
        return {"status": "ok", "model_loaded": True, "threshold": det.threshold}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "model_unavailable", "detail": str(e)},
        )


# ------------------------------------------------------------------
# Uploaded video file analysis
# ------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    det = get_detector()
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        timeline = det.score_video_file(save_path)
    finally:
        os.remove(save_path)

    anomaly_count = sum(1 for r in timeline if r["is_anomaly"])
    return {
        "filename": file.filename,
        "clips_analyzed": len(timeline),
        "anomalies_detected": anomaly_count,
        "timeline": timeline,
    }


# ------------------------------------------------------------------
# Webcam streaming via WebSocket
# ------------------------------------------------------------------
@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    det = get_detector()
    det.reset_stream()
    try:
        while True:
            data = await websocket.receive_text()
            # Expect a data URL: "data:image/jpeg;base64,...."
            header, b64data = data.split(",", 1) if "," in data else ("", data)
            jpg_bytes = base64.b64decode(b64data)
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = det.push_frame(frame)
            if result is None:
                await websocket.send_json({"status": "buffering"})
                continue

            await websocket.send_json({
                "status": "ok",
                "score": result["score"],
                "threshold": result["threshold"],
                "is_anomaly": result["is_anomaly"],
            })
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------
# RTSP stream monitoring (background thread)
# ------------------------------------------------------------------
class RTSPRequest(BaseModel):
    url: str


class _RTSPMonitor:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.latest: dict = {"status": "stopped"}

    def start(self, url: str):
        if self.running:
            self.stop()
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(url,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.latest = {"status": "stopped"}

    def _run(self, url: str):
        det = get_detector()
        det.reset_stream()
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            self.latest = {"status": "error", "detail": f"Could not open RTSP stream: {url}"}
            self.running = False
            return

        self.latest = {"status": "connected"}
        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.latest = {"status": "error", "detail": "Stream read failed"}
                time.sleep(1)
                continue
            result = det.push_frame(frame)
            if result is not None:
                self.latest = {
                    "status": "ok",
                    "score": result["score"],
                    "threshold": result["threshold"],
                    "is_anomaly": result["is_anomaly"],
                    "timestamp": time.time(),
                }
        cap.release()


_rtsp_monitor = _RTSPMonitor()


@app.post("/api/rtsp/start")
def rtsp_start(req: RTSPRequest):
    _rtsp_monitor.start(req.url)
    return {"status": "starting", "url": req.url}


@app.post("/api/rtsp/stop")
def rtsp_stop():
    _rtsp_monitor.stop()
    return {"status": "stopped"}


@app.get("/api/rtsp/status")
def rtsp_status():
    return _rtsp_monitor.latest


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
