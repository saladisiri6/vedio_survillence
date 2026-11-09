"""
Data loading & preprocessing for the surveillance anomaly detector.

Supports two sources:

1. UCSD Ped2 (the standard benchmark for this architecture) -- expects the
   dataset already downloaded and extracted locally, in its native layout:

       data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001/*.tif
       data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train002/*.tif
       ...

   Download it yourself from the official UCSD source (search
   "UCSD Anomaly Detection Dataset") since this environment cannot reach
   arbitrary hosting sites -- then point --data_dir at the extracted folder.

2. A synthetic fallback generator, used to sanity-check the full pipeline
   (shapes, training loop, inference, dashboard) without any real dataset.
   It draws moving-blob "normal" clips and occasionally injects a fast /
   erratic "anomalous" blob so you can confirm the anomaly scoring logic
   actually works end-to-end before pointing training at real footage.
"""

import glob
import os

import numpy as np
from PIL import Image

from model import SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH


def _load_frame(path: str, height: int, width: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((width, height))
    return np.asarray(img, dtype=np.float32) / 255.0


def load_ucsd_clips(
    data_dir: str,
    split: str = "Train",
    seq_len: int = SEQUENCE_LENGTH,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
    stride: int = 1,
) -> np.ndarray:
    """Loads UCSD Ped2 frames and slices them into overlapping clips.

    Returns an array of shape (N, seq_len, height, width, 1).
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Expected UCSD split directory at {split_dir}. "
            "Download & extract UCSD Ped2 first (see dataset.py docstring)."
        )

    clips = []
    for seq_folder in sorted(glob.glob(os.path.join(split_dir, "*"))):
        if not os.path.isdir(seq_folder):
            continue
        frame_paths = sorted(
            glob.glob(os.path.join(seq_folder, "*.tif"))
            + glob.glob(os.path.join(seq_folder, "*.jpg"))
            + glob.glob(os.path.join(seq_folder, "*.png"))
        )
        if len(frame_paths) < seq_len:
            continue
        frames = np.stack([_load_frame(p, height, width) for p in frame_paths])
        for start in range(0, len(frames) - seq_len + 1, stride):
            clip = frames[start : start + seq_len]
            clips.append(clip[..., np.newaxis])

    if not clips:
        raise RuntimeError(f"No clips found under {split_dir}")

    return np.stack(clips).astype(np.float32)


def generate_synthetic_clips(
    n_clips: int = 200,
    seq_len: int = SEQUENCE_LENGTH,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
    anomaly_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic moving-blob clips for pipeline sanity checks.

    Normal clips: a single blob drifting smoothly in a random direction.
    Anomalous clips (only if anomaly_ratio > 0): blob moves erratically /
    much faster, simulating unusual motion.

    Returns (clips, labels) where labels[i] == 1 means anomalous.
    """
    rng = np.random.default_rng(seed)
    clips = np.zeros((n_clips, seq_len, height, width, 1), dtype=np.float32)
    labels = np.zeros(n_clips, dtype=np.int32)

    yy, xx = np.mgrid[0:height, 0:width]

    for i in range(n_clips):
        is_anomaly = rng.random() < anomaly_ratio
        labels[i] = int(is_anomaly)

        cx, cy = rng.uniform(20, width - 20), rng.uniform(20, height - 20)
        if is_anomaly:
            vx, vy = rng.uniform(-15, 15, size=2)  # fast/erratic
            radius = rng.uniform(4, 8)
        else:
            vx, vy = rng.uniform(-3, 3, size=2)  # smooth drift
            radius = rng.uniform(6, 10)

        for t in range(seq_len):
            if is_anomaly and rng.random() < 0.3:
                # erratic jitter
                cx += rng.uniform(-10, 10)
                cy += rng.uniform(-10, 10)
            else:
                cx += vx
                cy += vy
            cx = np.clip(cx, 10, width - 10)
            cy = np.clip(cy, 10, height - 10)

            blob = np.exp(-(((xx - cy) ** 2 + (yy - cx) ** 2) / (2 * radius**2)))
            clips[i, t, :, :, 0] = blob

    return clips, labels


if __name__ == "__main__":
    clips, labels = generate_synthetic_clips(n_clips=8, anomaly_ratio=0.5)
    print("clips shape:", clips.shape, "labels:", labels)
