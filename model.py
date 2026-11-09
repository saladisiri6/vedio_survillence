"""
Spatio-Temporal AutoEncoder for video anomaly detection.

Architecture (based on Hasan et al. / Luo et al. convolutional-LSTM autoencoder
literature for surveillance anomaly detection):

    Input: (batch, T=10, H=128, W=128, C=1)  -- 10-frame grayscale clip

    Spatial Encoder (Conv3D)      -> compresses each frame spatially
    Temporal Encoder (ConvLSTM2D) -> learns normal motion/appearance patterns
    Temporal Decoder (ConvLSTM2D) -> reconstructs the temporal sequence
    Spatial Decoder (Conv3DTranspose) -> reconstructs full-resolution frames

The model is trained ONLY on normal (non-anomalous) footage. At inference
time, frames that reconstruct poorly (high MSE) are flagged as anomalies,
since the network never learned to reconstruct abnormal motion/appearance.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# Clip length (frames) and frame size the whole pipeline is built around.
SEQUENCE_LENGTH = 10
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
CHANNELS = 1


def build_autoencoder(
    seq_len: int = SEQUENCE_LENGTH,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
    channels: int = CHANNELS,
) -> tf.keras.Model:
    """Builds the Conv3D + ConvLSTM2D spatio-temporal autoencoder."""

    inputs = layers.Input(shape=(seq_len, height, width, channels), name="clip_input")

    # ---------------- Spatial Encoder (Conv3D) ----------------
    x = layers.Conv3D(
        128, kernel_size=(3, 11, 11), strides=(1, 4, 4), padding="same"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(
        64, kernel_size=(3, 5, 5), strides=(1, 2, 2), padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # shape now: (batch, T, H/8, W/8, 64)

    # ---------------- Temporal Encoder (ConvLSTM2D) ----------------
    x = layers.ConvLSTM2D(
        64, kernel_size=(3, 3), padding="same", return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
        32, kernel_size=(3, 3), padding="same", return_sequences=True
    )(x)
    encoded = layers.BatchNormalization()(x)

    # ---------------- Temporal Decoder (ConvLSTM2D) ----------------
    x = layers.ConvLSTM2D(
        64, kernel_size=(3, 3), padding="same", return_sequences=True
    )(encoded)
    x = layers.BatchNormalization()(x)

    # ---------------- Spatial Decoder (Conv3DTranspose) ----------------
    x = layers.Conv3DTranspose(
        64, kernel_size=(3, 5, 5), strides=(1, 2, 2), padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3DTranspose(
        128, kernel_size=(3, 11, 11), strides=(1, 4, 4), padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    outputs = layers.Conv3D(
        channels, kernel_size=(3, 3, 3), padding="same", activation="sigmoid",
        name="reconstruction",
    )(x)

    model = models.Model(inputs, outputs, name="spatio_temporal_autoencoder")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


if __name__ == "__main__":
    m = build_autoencoder()
    m.summary()
