"""Training script (Section 3.1.6): Adam optimizer (lr=0.001, beta1=0.9,
beta2=0.999), categorical cross-entropy with class weighting.

Supports --resume for Colab: free-tier sessions can disconnect mid-run,
and ModelCheckpoint below already saves the best weights to Drive as
training progresses, so a disconnect doesn't have to mean starting over
from random weights. --resume loads whatever was last checkpointed and
continues from there instead.

Usage:
    python train.py                # fresh run
    python train.py --resume       # continue from the last checkpoint
"""

import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from model import build_hybrid_model

CHECKPOINT_PATH = config.MODELS_DIR / "hybrid_cnn_rnn_best.keras"
# Tracks how many epochs have already been completed, so a resumed run
# doesn't reset EarlyStopping's patience counter or Keras's own epoch
# numbering back to 0 -- both would behave as if training just started,
# even though the model itself already has real progress.
PROGRESS_PATH = config.MODELS_DIR / "hybrid_training_progress.json"


def load_split(split_name):
    mel = np.load(config.SPLITS_DIR / f"{split_name}_mel.npy")
    mfcc = np.load(config.SPLITS_DIR / f"{split_name}_mfcc.npy")
    y = np.load(config.SPLITS_DIR / f"{split_name}_y.npy")
    return mel, mfcc, y


def main():
    parser = argparse.ArgumentParser(description="Train the hybrid CNN-RNN model.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Load the last checkpoint and continue training from there, "
             "instead of starting from random weights.",
    )
    args = parser.parse_args()

    train_mel, train_mfcc, y_train = load_split("train")
    val_mel, val_mfcc, y_val = load_split("val")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights_arr))
    print("Class weights:", class_weights)

    initial_epoch = 0

    if args.resume:
        if not CHECKPOINT_PATH.exists():
            print(
                f"--resume was given but no checkpoint found at "
                f"{CHECKPOINT_PATH} -- starting fresh instead."
            )
            model = build_hybrid_model(
                mel_shape=train_mel.shape[1:], mfcc_shape=train_mfcc.shape[1:]
            )
        else:
            print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
            model = tf.keras.models.load_model(CHECKPOINT_PATH)
            if PROGRESS_PATH.exists():
                with open(PROGRESS_PATH) as f:
                    initial_epoch = json.load(f)["completed_epochs"]
                print(f"Resuming from epoch {initial_epoch}")
            else:
                print(
                    "No progress file found alongside the checkpoint -- "
                    "resuming weights, but epoch count restarts at 0. "
                    "EarlyStopping's patience counter will also restart, "
                    "which is a minor inefficiency, not a correctness issue."
                )
    else:
        model = build_hybrid_model(
            mel_shape=train_mel.shape[1:], mfcc_shape=train_mfcc.shape[1:]
        )

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
            filepath=str(CHECKPOINT_PATH),
            monitor="val_loss", save_best_only=True,
        ),
        # Writes completed-epoch count after every epoch, so a disconnect
        # at any point still leaves an accurate progress file behind for
        # the next --resume to pick up from.
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: json.dump(
                {"completed_epochs": epoch + 1}, open(PROGRESS_PATH, "w")
            )
        ),
    ]

    model.fit(
        [train_mel, train_mfcc], y_train_cat,
        validation_data=([val_mel, val_mfcc], y_val_cat),
        epochs=config.EPOCHS,
        initial_epoch=initial_epoch,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(config.MODELS_DIR / "hybrid_cnn_rnn_final.keras")
    print("Training complete. Model saved to", config.MODELS_DIR)


if __name__ == "__main__":
    main()

