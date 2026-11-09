"""
Train the spatio-temporal autoencoder.

Usage:
    # Train on real UCSD Ped2 data (recommended):
    python train.py --data_dir data/UCSD_Anomaly_Dataset.v1p2/UCSDped2 --epochs 30

    # Sanity-check the whole pipeline with synthetic data (no dataset needed):
    python train.py --synthetic --epochs 5

The model is trained ONLY on normal footage (reconstruction-error approach):
low loss on frames similar to training data, high loss (= flagged anomaly)
on frames unlike anything seen during training.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from model import build_autoencoder, compile_model, SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH
from dataset import load_ucsd_clips, generate_synthetic_clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to UCSD Ped2 folder")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="../models/autoencoder.keras")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from ../models/checkpoint.keras if it exists, instead of starting fresh",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("[train] Using synthetic normal-only clips for a pipeline sanity check.")
        clips, _ = generate_synthetic_clips(n_clips=300, anomaly_ratio=0.0)
    else:
        if not args.data_dir:
            raise SystemExit("Pass --data_dir <path to UCSDped2> or use --synthetic")
        print(f"[train] Loading UCSD clips from {args.data_dir} ...")
        clips = load_ucsd_clips(args.data_dir, split="Train")

    print(f"[train] Loaded {clips.shape[0]} clips of shape {clips.shape[1:]}")

    n_val = int(len(clips) * args.val_split)
    idx = np.random.default_rng(0).permutation(len(clips))
    val_clips = clips[idx[:n_val]]
    train_clips = clips[idx[n_val:]]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    checkpoint_path = os.path.join(os.path.dirname(args.out) or ".", "checkpoint.keras")

    if args.resume and os.path.exists(checkpoint_path):
        print(f"[train] --resume passed and checkpoint found: loading {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        compile_model(model)
    else:
        if args.resume:
            print("[train] --resume passed but no checkpoint found yet -- starting fresh.")
        model = build_autoencoder(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH)
        compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        # Saved after EVERY epoch, so Ctrl+C or a crash never loses more than
        # one epoch of progress. Re-run with --resume to pick back up here.
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_freq="epoch", save_weights_only=False,
        ),
    ]

    model.fit(
        train_clips, train_clips,  # autoencoder: input == target
        validation_data=(val_clips, val_clips) if n_val > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"[train] Saved model to {args.out}")

    # Save a reconstruction-error threshold derived from the training set itself,
    # used later at inference time to flag anomalies.
    recon = model.predict(train_clips, batch_size=args.batch_size)
    errors = np.mean((recon - train_clips) ** 2, axis=(1, 2, 3, 4))
    threshold = float(np.mean(errors) + 3 * np.std(errors))
    threshold_path = os.path.join(os.path.dirname(args.out), "threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    print(f"[train] Anomaly threshold (mean + 3*std of train recon error): {threshold:.6f}")
    print(f"[train] Saved threshold to {threshold_path}")


if __name__ == "__main__":
    main()
