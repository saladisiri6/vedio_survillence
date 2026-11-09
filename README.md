# SENTINEL — Intelligent Video Surveillance (Anomaly Detection)

A production-style, deep-learning video anomaly detector using a
**Spatio-Temporal AutoEncoder (Conv3D + ConvLSTM2D)**, served through a
**FastAPI** backend with a custom HTML/JS dashboard supporting **file upload,
webcam, and RTSP** input.

The model is unsupervised: it's trained only on *normal* footage and learns
to reconstruct it well. At inference time, any clip the model can't
reconstruct well (high MSE) gets flagged as an anomaly — no labeled
abnormal-event data required.

## How it works

```
raw frames -> grayscale, resize to 128x128, stack into 10-frame clips
           -> Conv3D spatial encoder (compress each frame)
           -> ConvLSTM2D temporal encoder/decoder (learn normal motion)
           -> Conv3DTranspose spatial decoder (reconstruct clip)
           -> reconstruction error (MSE) per clip
           -> error > threshold  =>  ANOMALY
```

The threshold is derived automatically after training: `mean + 3*std` of
reconstruction error across the training set.

## Project layout

```
video_surveillance/
├── app/
│   ├── model.py       # Conv3D + ConvLSTM2D autoencoder architecture
│   ├── dataset.py      # UCSD Ped2 loader + synthetic data generator
│   ├── train.py        # training script (saves model + threshold)
│   ├── inference.py     # AnomalyDetector: streaming + batch scoring
│   └── main.py          # FastAPI app (upload / webcam / RTSP endpoints)
├── static/
│   ├── index.html       # dashboard UI
│   ├── style.css        # CCTV/phosphor-monitor themed styling
│   └── app.js            # upload, webcam WS client, RTSP polling, waveform
├── models/               # trained model + threshold.txt land here
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Get training data

This architecture is built around the standard **UCSD Ped2** benchmark.
Download it yourself (search "UCSD Anomaly Detection Dataset", official UCSD
CVRR page) and extract it so you have:

```
data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001/*.tif
data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train002/*.tif
...
```

Don't have the dataset yet, or just want to confirm the whole pipeline
(shapes, training loop, dashboard) works first? Use the built-in synthetic
generator — no download needed:

```bash
cd app
python train.py --synthetic --epochs 5
```

## 2. Train

```bash
cd app
python train.py --data_dir ../data/UCSD_Anomaly_Dataset.v1p2/UCSDped2 --epochs 30
```

This saves `../models/autoencoder.keras` and `../models/threshold.txt`.
Training on the full-resolution model is CPU-heavy — a GPU is strongly
recommended (Colab, a local CUDA box, or a cloud GPU instance).

## 3. Run the app

```bash
cd app
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — the SENTINEL dashboard.

- **File Upload** — drop a video, get a clip-by-clip anomaly timeline.
- **Webcam** — streams your browser's webcam over a WebSocket to the model
  for live scoring (buffers a 10-frame sliding window, then scores continuously).
- **RTSP** — give it a stream URL (`rtsp://user:pass@host:554/stream`); the
  server pulls frames in a background thread and you poll for live results.

## Notes / things to tune for your footage

- `SEQUENCE_LENGTH`, `FRAME_HEIGHT`, `FRAME_WIDTH` in `model.py` control clip
  length and resolution — keep them consistent between training and inference.
- The auto-derived threshold (`mean + 3*std`) is a reasonable starting point,
  but you'll likely want to tune it against a small labeled validation set
  from your own footage/camera to balance false positives vs. missed events.
- For RTSP in production, consider running the frame-reading loop in a
  separate process (not just a thread) if your camera has high latency or
  drops connections often — the current implementation uses a daemon thread
  for simplicity.
