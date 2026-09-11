"""Grad-CAM explainability (Section 3.2.1 / FUN-07).

The thesis originally listed LRP as an option alongside Grad-CAM, but LRP
in Python is best supported through Captum, which is PyTorch-only -- and
this whole stack is TensorFlow/Keras. Grad-CAM is natively supported in
TensorFlow via tf.GradientTape, so that's what's implemented here; this
matches the resolved decision already noted in the project (Section
3.2.1 mentions both, but Grad-CAM is the one that fits a TF stack).

Grad-CAM only works on the CNN branch (it needs a convolutional layer to
localize over), so it explains which time-frequency regions of the
Mel-spectrogram most influenced the "fake" prediction. It doesn't produce
an explanation for the RNN/MFCC branch -- LSTMs don't have the spatial
structure Grad-CAM relies on, and that's a normal limitation, not a bug.

Works on:
  - the hybrid model (mel_input branch)
  - the cnn_only baseline
It will not work on the rnn_only baseline, since there's no conv layer.
"""

import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config


def find_last_conv_layer(model):
    """Searches backward through the model's layers for the last Conv2D,
    which is standard practice for Grad-CAM (the last conv layer captures
    the highest-level spatial features before pooling/flattening)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(
        "No Conv2D layer found in this model -- Grad-CAM needs a "
        "convolutional layer to localize over (e.g. the CNN branch of "
        "the hybrid model, or the cnn_only baseline)."
    )


def compute_gradcam(model, model_inputs, class_index, conv_layer_name=None):
    """Runs Grad-CAM for one sample and returns a 2D heatmap (values in
    [0, 1]) sized to the conv layer's spatial dimensions.

    model_inputs: whatever the model expects for a single sample, wrapped
    in a batch dimension of 1 -- e.g. [mel_batch1, mfcc_batch1] for the
    hybrid model, or just mel_batch1 for the cnn_only baseline.
    class_index: which output class to explain (0 = real, 1 = fake).
    """
    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(model_inputs)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    # Global-average-pool the gradients over the spatial dims -> one
    # importance weight per channel, the standard Grad-CAM weighting.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: we only care about features that positively influenced this
    # class, not ones that pushed away from it.
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_mel(mel_spectrogram, heatmap, save_path, title=None):
    """Resizes the (small) Grad-CAM heatmap up to the Mel-spectrogram's
    shape and saves a side-by-side + overlay figure.

    mel_spectrogram: (time, n_mels) array, as produced by features.py.
    heatmap: 2D array from compute_gradcam(), smaller spatial size.
    """
    mel_img = mel_spectrogram.T  # -> (n_mels, time) for a natural plot orientation

    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (mel_img.shape[0], mel_img.shape[1])
    ).numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(mel_img, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Mel-spectrogram")
    axes[0].set_xlabel("Time frame")
    axes[0].set_ylabel("Mel band")

    axes[1].imshow(mel_img, aspect="auto", origin="lower", cmap="magma")
    axes[1].imshow(heatmap_resized, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM overlay")
    axes[1].set_xlabel("Time frame")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def explain_prediction(model, mel_sample, mfcc_sample=None, save_path=None, sample_label=None):
    """Convenience wrapper: takes one raw sample (no batch dim), runs a
    forward pass to get the predicted class, computes Grad-CAM for that
    class, and saves an overlay figure.

    mfcc_sample is only needed for the hybrid model; pass None for the
    cnn_only baseline.
    """
    mel_batch = mel_sample[np.newaxis, ...]

    if mfcc_sample is not None:
        mfcc_batch = mfcc_sample[np.newaxis, ...]
        model_inputs = [mel_batch, mfcc_batch]
    else:
        model_inputs = mel_batch

    probs = model.predict(model_inputs, verbose=0)
    predicted_class = int(np.argmax(probs[0]))
    confidence = float(probs[0][predicted_class])

    heatmap = compute_gradcam(model, model_inputs, predicted_class)

    if save_path is None:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = config.RESULTS_DIR / "gradcam_example.png"

    label_str = "fake" if predicted_class == 1 else "real"
    title = f"Predicted: {label_str} (confidence {confidence:.2f})"
    if sample_label is not None:
        true_str = "fake" if sample_label == 1 else "real"
        title += f" | True: {true_str}"

    overlay_heatmap_on_mel(mel_sample, heatmap, save_path, title=title)
    print(f"Saved Grad-CAM overlay to {save_path}")
    return heatmap, predicted_class, confidence


def batch_explain_predictions(
    model, mel, mfcc=None, y_true=None, n_per_category=3, output_dir=None,
):
    """Generates Grad-CAM overlays across several test samples instead of
    just one, which is what you actually want for a thesis figure set --
    a single example doesn't tell you whether the model generally attends
    to sensible regions or whether that one image was a fluke.

    If y_true is given, samples are drawn from four prediction categories
    (true positive, false positive, true negative, false negative) so you
    can visually compare what the model looks at when it's right vs. wrong
    -- a standard way to present Grad-CAM results and a natural fit for
    Section 3.2.1's explainability discussion. If y_true is None, samples
    are just drawn in order from the start of the array.

    mfcc is only needed for the hybrid model; pass None for the cnn_only
    baseline (mirrors explain_prediction()).

    Returns a list of dicts, one per saved image, with the sample index,
    predicted class, true label (if known), and confidence -- handy for
    building a caption table alongside the figures.
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "gradcam_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run predictions once for the whole batch so we know which samples
    # fall into which category, rather than predicting one-by-one inside
    # the sampling loop below.
    if mfcc is not None:
        all_probs = model.predict([mel, mfcc], verbose=0)
    else:
        all_probs = model.predict(mel, verbose=0)
    all_preds = np.argmax(all_probs, axis=1)

    if y_true is not None:
        y_true = np.asarray(y_true)
        categories = {
            "true_positive": np.where((all_preds == 1) & (y_true == 1))[0],
            "false_positive": np.where((all_preds == 1) & (y_true == 0))[0],
            "true_negative": np.where((all_preds == 0) & (y_true == 0))[0],
            "false_negative": np.where((all_preds == 0) & (y_true == 1))[0],
        }
    else:
        # No ground truth available -- just take the first N samples as
        # one unlabeled group.
        categories = {"sample": np.arange(len(mel))}

    manifest = []
    for category_name, indices in categories.items():
        if len(indices) == 0:
            print(f"No samples found for category '{category_name}' -- skipping.")
            continue

        chosen = indices[:n_per_category]
        for rank, idx in enumerate(chosen):
            idx = int(idx)
            mel_sample = mel[idx]
            mfcc_sample = mfcc[idx] if mfcc is not None else None
            true_label = int(y_true[idx]) if y_true is not None else None

            save_path = output_dir / f"{category_name}_{rank}_idx{idx}.png"
            heatmap, predicted_class, confidence = explain_prediction(
                model, mel_sample, mfcc_sample,
                save_path=save_path, sample_label=true_label,
            )

            manifest.append({
                "category": category_name,
                "sample_index": idx,
                "predicted_class": predicted_class,
                "true_label": true_label,
                "confidence": confidence,
                "image_path": str(save_path),
            })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {len(manifest)} Grad-CAM overlays to {output_dir}")
    print(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    # Quick manual smoke test: explain a handful of test-split samples,
    # spread across TP/FP/TN/FN, using the hybrid model, if it and the
    # test split both exist.
    model_path = config.MODELS_DIR / "hybrid_cnn_rnn_best.keras"
    if not model_path.exists():
        print(f"No saved model found at {model_path} -- train one first.")
    else:
        mel = np.load(config.SPLITS_DIR / "test_mel.npy")
        mfcc = np.load(config.SPLITS_DIR / "test_mfcc.npy")
        y = np.load(config.SPLITS_DIR / "test_y.npy")

        model = tf.keras.models.load_model(model_path)
        batch_explain_predictions(model, mel, mfcc, y_true=y, n_per_category=3)
