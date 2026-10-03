"""
fit_temperature.py — Post-hoc calibration (temperature scaling) for the
hybrid CNN+BiLSTM deepfake detector, per Guo et al. (2017), "On
Calibration of Modern Neural Networks."

WHY: your softmax output is baked directly into the final Dense layer
(activation="softmax" in model.py), which tends to produce very
overconfident probabilities (e.g. 100.0%) even when the model shouldn't
be that certain. Temperature scaling fixes this WITHOUT retraining and
WITHOUT changing which class is predicted (it doesn't touch argmax) --
it just makes the confidence numbers more honest.

WHAT IT DOES:
  1. Rebuilds the trained model's architecture with the final layer's
     softmax removed (so we get raw logits), copying over the trained
     weights -- since activation doesn't affect weight shapes, this is
     a safe, lossless operation, not a retrain.
  2. Fits a single scalar T on the VALIDATION split (never the test
     split -- fitting on test would leak test information into your
     reported metrics) such that softmax(logits / T) is better
     calibrated.
  3. Saves T to results/temperature.json for use in app.py.

Run this ONCE after training (or after retraining). You do not need to
re-run it every time you launch the Streamlit app -- app.py just reads
the saved T.

Usage:
    python fit_temperature.py
"""

import json

import numpy as np
import tensorflow as tf

import config

MODEL_PATH = config.MODELS_DIR / "hybrid_cnn_rnn_best.keras"

# Different projects name their validation split differently (val_*,
# dev_*, validation_*) -- try the common conventions in order rather
# than requiring you to confirm which one this project uses.
SPLIT_NAME_CANDIDATES = ["val", "dev", "validation"]


def find_split_files():
    for name in SPLIT_NAME_CANDIDATES:
        mel_path = config.SPLITS_DIR / f"{name}_mel.npy"
        mfcc_path = config.SPLITS_DIR / f"{name}_mfcc.npy"
        y_path = config.SPLITS_DIR / f"{name}_y.npy"
        if mel_path.exists() and mfcc_path.exists() and y_path.exists():
            print(f"Using validation split: '{name}_*.npy'")
            return mel_path, mfcc_path, y_path
    raise FileNotFoundError(
        f"Couldn't find a validation split in {config.SPLITS_DIR}. "
        f"Tried prefixes: {SPLIT_NAME_CANDIDATES}. "
        f"Files actually present: {[p.name for p in config.SPLITS_DIR.glob('*.npy')]}\n"
        f"If your validation split uses a different prefix, add it to "
        f"SPLIT_NAME_CANDIDATES above and re-run."
    )


def build_logits_model(trained_model):
    """Clones the model's architecture, swapping the final
    'classification' Dense layer's activation from softmax to linear,
    then copies over the trained weights. This gives pre-softmax logits
    without retraining -- weight shapes are identical regardless of
    activation, so set_weights() is exact."""
    def clone_fn(layer):
        cfg = layer.get_config()
        if layer.name == "classification":
            cfg = dict(cfg)
            cfg["activation"] = "linear"
            return layer.__class__.from_config(cfg)
        return layer.__class__.from_config(layer.get_config())

    logits_model = tf.keras.models.clone_model(trained_model, clone_function=clone_fn)
    logits_model.set_weights(trained_model.get_weights())
    return logits_model


def fit_temperature(logits, y_true, n_steps=500, lr=0.01):
    """Finds the scalar T minimizing NLL of softmax(logits / T) against
    y_true. T is kept positive via a softplus reparam (standard
    temperature-scaling implementation) so the optimizer can't wander
    into an invalid T <= 0."""
    logits = tf.constant(logits, dtype=tf.float32)
    y_true = tf.constant(y_true, dtype=tf.int32)

    raw_T = tf.Variable(0.0)  # softplus(0) ~= 0.69, a reasonable start
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            T = tf.nn.softplus(raw_T) + 1e-3
            scaled = logits / T
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_true, scaled, from_logits=True)
            )
        grads = tape.gradient(loss, [raw_T])
        optimizer.apply_gradients(zip(grads, [raw_T]))

    return float(tf.nn.softplus(raw_T) + 1e-3)


def main():
    mel_path, mfcc_path, y_path = find_split_files()
    mel = np.load(mel_path, mmap_mode="r")
    mfcc = np.load(mfcc_path, mmap_mode="r")
    y_true = np.load(y_path, mmap_mode="r")

    trained_model = tf.keras.models.load_model(MODEL_PATH)
    logits_model = build_logits_model(trained_model)

    logits = logits_model.predict([mel, mfcc], batch_size=config.BATCH_SIZE)

    # Before/after so you can see + report the calibration effect directly
    probs_before = tf.nn.softmax(logits).numpy()
    confidence_before = float(np.mean(np.max(probs_before, axis=1)))

    T = fit_temperature(logits, y_true)

    probs_after = tf.nn.softmax(logits / T).numpy()
    confidence_after = float(np.mean(np.max(probs_after, axis=1)))

    print(f"\nFitted temperature T = {T:.4f}")
    print(f"Mean top-class confidence BEFORE scaling: {confidence_before * 100:.1f}%")
    print(f"Mean top-class confidence AFTER  scaling: {confidence_after * 100:.1f}%")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "temperature.json"
    with open(out_path, "w") as f:
        json.dump({"T": T}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
