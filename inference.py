"""
Inference utilities: turn a raw video (or live frame buffer) into
per-clip anomaly scores using the trained spatio-temporal autoencoder.
"""

import os
from collections import deque
from typing import Optional

import numpy as np
import cv2
import tensorflow as tf

from model import SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH


class AnomalyDetector:
    def __init__(
        self,
        model_path: str,
        threshold: Optional[float] = None,
        seq_len: int = SEQUENCE_LENGTH,
        height: int = FRAME_HEIGHT,
        width: int = FRAME_WIDTH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run train.py first."
            )
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.seq_len = seq_len
        self.height = height
        self.width = width

        if threshold is None:
            threshold_path = os.path.join(os.path.dirname(model_path), "threshold.txt")
            if os.path.exists(threshold_path):
                with open(threshold_path) as f:
                    threshold = float(f.read().strip())
            else:
                threshold = 0.02  # sane fallback default
        self.threshold = threshold

        # Rolling buffer for streaming (webcam / RTSP) use.
        self._buffer: deque = deque(maxlen=seq_len)

    # ---------- frame preprocessing ----------
    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return (resized.astype(np.float32) / 255.0)[..., np.newaxis]

    def _score_clip(self, clip: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns (scalar anomaly score, per-pixel error map for the last frame)."""
        batch = clip[np.newaxis, ...]  # (1, T, H, W, 1)
        recon = self.model.predict(batch, verbose=0)[0]
        error_map = (recon - clip) ** 2  # (T, H, W, 1)
        score = float(np.mean(error_map))
        last_frame_error = error_map[-1, :, :, 0]
        return score, last_frame_error

    # ---------- streaming API (webcam / RTSP) ----------
    def push_frame(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Feed one live frame. Returns a result dict once enough frames are
        buffered (every call after that re-scores the sliding window)."""
        self._buffer.append(self._preprocess_frame(frame_bgr))
        if len(self._buffer) < self.seq_len:
            return None
        clip = np.stack(self._buffer, axis=0)
        score, error_map = self._score_clip(clip)
        return {
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": score > self.threshold,
            "error_map": error_map,  # caller can render as heatmap overlay
        }

    def reset_stream(self):
        self._buffer.clear()

    # ---------- batch API (uploaded video file) ----------
    def score_video_file(self, video_path: str, stride: int = 5) -> list[dict]:
        """Scores an uploaded video file clip-by-clip (sliding window with
        the given stride) and returns a timeline of results."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(self._preprocess_frame(frame))
        cap.release()

        results = []
        for start in range(0, max(1, len(frames) - self.seq_len + 1), stride):
            clip_frames = frames[start : start + self.seq_len]
            if len(clip_frames) < self.seq_len:
                break
            clip = np.stack(clip_frames, axis=0)
            score, _ = self._score_clip(clip)
            results.append({
                "start_frame": start,
                "timestamp_sec": round(start / fps, 2),
                "score": score,
                "threshold": self.threshold,
                "is_anomaly": score > self.threshold,
            })
        return results
