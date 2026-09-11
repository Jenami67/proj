"""Trains the CNN-only and RNN-only baseline models (needed to answer
Research Question 3: how does the hybrid architecture compare to CNN-only
or RNN-only models?).

This mirrors train.py's setup (same optimizer, class weighting, callbacks)
but is kept as a separate script rather than folded into train.py, since
each baseline only needs one feature array as input instead of two, and
keeping them separate avoids branching logic inside train.py that would
make it harder to follow for a single model.

Usage:
    python train_baselines.py --model cnn_only
    python train_baselines.py --model rnn_only
    python train_baselines.py --model both      # trains both, one after another
"""

import argparse

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from model import build_cnn_only_model, build_rnn_only_model


def load_split(split_name):
    mel = np.load(config.SPLITS_DIR / f"{split_name}_mel.npy")
    mfcc = np.load(config.SPLITS_DIR / f"{split_name}_mfcc.npy")
    y = np.load(config.SPLITS_DIR / f"{split_name}_y.npy")
    return mel, mfcc, y


def train_one_baseline(model_kind):
    """model_kind: 'cnn_only' or 'rnn_only'."""
    train_mel, train_mfcc, y_train = load_split("train")
    val_mel, val_mfcc, y_val = load_split("val")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"\n[{model_kind}] Class weights:", class_weights)

    if model_kind == "cnn_only":
        model = build_cnn_only_model(mel_shape=train_mel.shape[1:])
        train_x, val_x = train_mel, val_mel
    elif model_kind == "rnn_only":
        model = build_rnn_only_model(mfcc_shape=train_mfcc.shape[1:])
        train_x, val_x = train_mfcc, val_mfcc
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE, beta_1=0.9, beta_2=0.999
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / f"{model_kind}_best.keras"),
            monitor="val_loss", save_best_only=True,
        ),
    ]

    model.fit(
        train_x, y_train_cat,
        validation_data=(val_x, y_val_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    final_path = config.MODELS_DIR / f"{model_kind}_final.keras"
    model.save(final_path)
    print(f"[{model_kind}] Training complete. Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN-only / RNN-only baselines.")
    parser.add_argument(
        "--model", choices=["cnn_only", "rnn_only", "both"], default="both",
        help="Which baseline(s) to train (default: both).",
    )
    args = parser.parse_args()

    kinds = ["cnn_only", "rnn_only"] if args.model == "both" else [args.model]
    for kind in kinds:
        train_one_baseline(kind)


if __name__ == "__main__":
    main()
