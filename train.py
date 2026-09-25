"""Training script (Section 3.1.6): Adam optimizer (lr=0.001, beta1=0.9,
beta2=0.999), categorical cross-entropy with class weighting.

Data loading uses tf.data.Dataset.from_generator() instead of passing raw
NumPy arrays straight to model.fit(). An earlier version of this script
passed full arrays directly to model.fit(), which makes Keras stage the
whole array for GPU transfer rather than streaming batches on demand --
fine for the dummy smoke-test data, but on the real ASVspoof train split
(train_mel.npy alone is ~4GB) that, plus TensorFlow's own graph/
activation memory, was enough to exhaust the Colab free-tier T4's VRAM.

A follow-up attempt used tf.data.Dataset.from_tensor_slices() on top of
mmap_mode='r' numpy arrays, expecting the memmap to avoid a full read --
but from_tensor_slices() converts its input into an in-memory EagerTensor
immediately when the dataset is built, which forces the *entire* memmapped
array to be read into RAM at that point regardless of the mmap. The "5
allocation exceeds 10% of free system memory" warnings in testing were
that conversion happening, not GPU/VRAM issue -- the fix never took
effect for training in memory-limited environments.

from_generator() is different: it wraps a Python generator function and
only calls it (pulling one sample at a time) when the pipeline actually
needs data, so indexing into the memmapped array happens lazily,
sample-by-sample, and only a batch's worth is ever materialized as a
real tensor at once.

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
    """mmap_mode='r' means these arrays are memory-mapped from disk, not
    fully read into RAM -- NumPy pages in only the slices that are
    actually indexed. This only pays off if whatever reads from the
    array also indexes it lazily -- see make_dataset() below, which
    uses a generator specifically so this mmap isn't defeated by an
    eager full-array conversion."""
    mel = np.load(config.SPLITS_DIR / f"{split_name}_mel.npy", mmap_mode="r")
    mfcc = np.load(config.SPLITS_DIR / f"{split_name}_mfcc.npy", mmap_mode="r")
    y = np.load(config.SPLITS_DIR / f"{split_name}_y.npy", mmap_mode="r")
    return mel, mfcc, y


def make_dataset(mel, mfcc, y_cat, batch_size, shuffle):
    """Builds a tf.data pipeline that streams (mel, mfcc) -> label
    batches without ever converting the full mel/mfcc arrays into an
    in-memory tensor. from_generator() calls the generator function
    lazily -- it only reads mel[i]/mfcc[i]/y_cat[i] from the memmapped
    arrays as each sample is actually needed by the pipeline, so at most
    a shuffle buffer's worth of samples plus the current batch exist as
    real tensors at any moment.
    """
    n_samples = mel.shape[0]
    mel_shape = mel.shape[1:]
    mfcc_shape = mfcc.shape[1:]

    def generator():
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng(config.RANDOM_STATE)
            rng.shuffle(indices)
        for i in indices:
            # np.asarray(...) copies just this one sample out of the
            # memmap into a small, ordinary in-memory array -- the rest
            # of the mapped file stays untouched on disk.
            yield (
                (np.asarray(mel[i], dtype=np.float32), np.asarray(mfcc[i], dtype=np.float32)),
                np.asarray(y_cat[i], dtype=np.float32),
            )

    output_signature = (
        (
            tf.TensorSpec(shape=mel_shape, dtype=tf.float32),
            tf.TensorSpec(shape=mfcc_shape, dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(y_cat.shape[1],), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


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

    train_dataset = make_dataset(
        train_mel, train_mfcc, y_train_cat, config.BATCH_SIZE, shuffle=True
    )
    val_dataset = make_dataset(
        val_mel, val_mfcc, y_val_cat, config.BATCH_SIZE, shuffle=False
    )

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
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(config.MODELS_DIR / "hybrid_cnn_rnn_final.keras")
    print("Training complete. Model saved to", config.MODELS_DIR)


if __name__ == "__main__":
    main()


