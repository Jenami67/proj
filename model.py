"""Hybrid CNN-RNN architecture (Section 3.2.1 / Figure 2).

Two-branch design, matching the block diagram:
  - CNN branch: analyzes the Mel-spectrogram for spectral texture patterns.
  - RNN branch (bidirectional LSTM): analyzes the MFCC sequence for
    temporal dependencies.
The branches are concatenated ("fusion") before the final dense +
classification layers.

Also includes CNN-only and RNN-only baselines, needed to answer Research
Question 3 ("how does the hybrid architecture compare to CNN-only or
RNN-only models").
"""

from tensorflow.keras import layers, models


def build_cnn_branch(mel_shape):
    inputs = layers.Input(shape=mel_shape, name="mel_input")
    x = layers.Reshape((mel_shape[0], mel_shape[1], 1))(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    return inputs, x


def build_rnn_branch(mfcc_shape):
    inputs = layers.Input(shape=mfcc_shape, name="mfcc_input")
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    return inputs, x


def build_hybrid_model(mel_shape, mfcc_shape, num_classes=2):
    cnn_input, cnn_features = build_cnn_branch(mel_shape)
    rnn_input, rnn_features = build_rnn_branch(mfcc_shape)

    fused = layers.Concatenate(name="fusion")([cnn_features, rnn_features])
    x = layers.Dense(128, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    return models.Model(
        inputs=[cnn_input, rnn_input], outputs=outputs,
        name="hybrid_cnn_rnn_deepfake_detector",
    )


def build_cnn_only_model(mel_shape, num_classes=2):
    inputs, x = build_cnn_branch(mel_shape)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="cnn_only_baseline")


def build_rnn_only_model(mfcc_shape, num_classes=2):
    inputs, x = build_rnn_branch(mfcc_shape)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="rnn_only_baseline")
