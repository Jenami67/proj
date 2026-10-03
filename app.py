"""
app.py — Deepfake Voice Detection Interface (Objective 5)
Wired exactly to your config.py / features.py / model.py.

  O5-T1: accepts an audio file as input
  O5-T2: extracts features + runs the trained hybrid CNN+BiLSTM model
  O5-T3: displays real vs deepfake result
  FUN-07: shows a Grad-CAM explanation heatmap (CNN / Mel-spectrogram branch)
  CAL-01: temperature-scaled confidence (Guo et al., 2017) — fixes the
          near-constant 100% confidence from the raw softmax output.
          Requires results/temperature.json (run fit_temperature.py once
          first). Falls back to unscaled (T=1.0) if that file is missing,
          so this never blocks the app from running.

RUN:
    streamlit run app.py

ONE THING TO CONFIRM (marked TODO below): the saved model filename in
models/. Label order (0=bonafide, 1=spoof) confirmed against
build_dataset.py's gather_files() docstring.
"""

import json
import os
import tempfile

import numpy as np
import librosa
import streamlit as st

import config
from features import extract_mfcc, extract_mel_spectrogram
from gradcam_utils import compute_gradcam, overlay_heatmap_on_mel

CLASS_NAMES = {0: "Bonafide (Real)", 1: "Spoof (Deepfake)"}  # confirmed vs build_dataset.py


@st.cache_resource
def load_trained_model():
    import tensorflow as tf
    # Your models/ folder also contains cnn_only_* and rnn_only_* baselines,
    # so we must target the hybrid model specifically (not just "first file
    # found"). Prefer the "best" checkpoint over "final".
    model_dir = config.MODELS_DIR
    preferred = [
        model_dir / "hybrid_cnn_rnn_best.keras",
        model_dir / "hybrid_cnn_rnn_final.keras",
    ]
    for path in preferred:
        if path.exists():
            model = tf.keras.models.load_model(str(path))
            logits_model = build_logits_model(model, tf)

            # Warm-up both models now so TF's graph tracing happens here
            # (once, at startup) instead of during the user's first click.
            dummy_frames = 1 + config.N_SAMPLES // config.HOP_LENGTH
            dummy_mel = np.zeros((1, dummy_frames, config.N_MELS), dtype=np.float32)
            dummy_mfcc = np.zeros((1, dummy_frames, config.N_MFCC * 3), dtype=np.float32)
            model([dummy_mel, dummy_mfcc], training=False)
            logits_model([dummy_mel, dummy_mfcc], training=False)

            return model, logits_model, str(path)

    raise FileNotFoundError(
        f"Couldn't find hybrid_cnn_rnn_best.keras or hybrid_cnn_rnn_final.keras "
        f"in {model_dir}. Files present: {[p.name for p in model_dir.glob('*.keras')]}"
    )


def build_logits_model(trained_model, tf):
    """Rebuilds the same architecture with the final 'classification'
    Dense layer's softmax swapped for linear, then copies over the
    trained weights — activation doesn't affect weight shapes, so this
    is exact, not a retrain. Needed because temperature scaling operates
    on pre-softmax logits, not on probabilities."""
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


@st.cache_resource
def load_temperature():
    """Reads the T fitted by fit_temperature.py on the validation split.
    Falls back to T=1.0 (no scaling) if it hasn't been run yet, so the
    app still works — just with the original overconfident numbers —
    rather than crashing."""
    t_path = config.RESULTS_DIR / "temperature.json"
    if t_path.exists():
        with open(t_path) as f:
            return json.load(f)["T"]
    return 1.0


def preprocess_audio(audio_path):
    """Load, fix to config.N_SAMPLES length, apply pre-emphasis — matching
    config.py's Table 3 settings. (Silence-trim logic from your
    audio_utils.py isn't included here since that file wasn't shared —
    send it if trimming matters for your real samples.)"""
    y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE)

    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]

    y = librosa.effects.preemphasis(y, coef=config.PRE_EMPHASIS_ALPHA)
    return y


def predict(logits_model, T, audio_path):
    import tensorflow as tf

    y = preprocess_audio(audio_path)

    mel = extract_mel_spectrogram(y)   # (time, 128)
    mfcc = extract_mfcc(y)             # (time, 39)

    mel_batch = mel[np.newaxis, ...]    # (1, time, 128)
    mfcc_batch = mfcc[np.newaxis, ...]  # (1, time, 39)

    # Direct call (not .predict()) — lower per-call overhead for a single
    # sample. Uses the logits model + temperature scaling instead of the
    # raw baked-in softmax, so confidence is calibrated rather than
    # near-constant 100%. This does NOT change which class wins — softmax
    # is monotonic, so argmax(logits) == argmax(logits / T) == argmax of
    # the original softmax output — only the confidence number changes.
    logits = logits_model([mel_batch, mfcc_batch], training=False).numpy()[0]
    probs = tf.nn.softmax(logits / T).numpy()

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence, mel, mfcc


def explain(model, mel, mfcc, predicted_class):
    """Grad-CAM heatmap for the predicted class, saved to a temp PNG.
    Uses the original softmax model (unchanged) — the predicted class is
    identical either way, so the explanation target doesn't change.
    Returns the PNG path (caller shows it, then deletes it)."""
    heatmap = compute_gradcam(
        model, [mel[np.newaxis, ...], mfcc[np.newaxis, ...]], predicted_class
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        png_path = f.name
    overlay_heatmap_on_mel(
        mel, heatmap, png_path, title=f"Explaining: {CLASS_NAMES[predicted_class]}"
    )
    return png_path


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Deepfake Voice Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Deepfake Voice Detection")
st.caption("An Intelligent Deep Learning Framework for Deepfake Voice Detection Using Spectral Feature Analysis")
st.markdown("Upload an audio clip and the trained CNN+BiLSTM model will classify it as **real** or **deepfake**.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("Run Detection", type="primary"):
        with st.spinner("Extracting features and running model..."):
            try:
                model, logits_model, model_path = load_trained_model()
                T = load_temperature()
                predicted_class, confidence, mel, mfcc = predict(logits_model, T, tmp_path)
                label = CLASS_NAMES[predicted_class]

                if predicted_class == 1:
                    st.error(f"### Result: {label}")
                else:
                    st.success(f"### Result: {label}")

                st.metric("Confidence", f"{confidence * 100:.1f}%")
                if T == 1.0:
                    st.caption(
                        "⚠️ Showing uncalibrated confidence — run fit_temperature.py "
                        "once to get calibrated numbers (results/temperature.json not found)."
                    )

                # Explanation heatmap
                st.subheader("Why this result?")
                png_path = explain(model, mel, mfcc, predicted_class)
                st.image(png_path)
                os.remove(png_path)
                st.caption(
                    "Grad-CAM: warmer colours mark the Mel-spectrogram regions "
                    "(time × frequency) that most influenced the predicted class. "
                    "This explains the CNN branch only, not the BiLSTM/MFCC branch."
                )

                with st.expander("Details"):
                    st.write(f"Model file used: `{model_path}`")
                    st.write(f"Audio fixed to {config.DURATION_SECONDS}s @ {config.SAMPLE_RATE}Hz")
                    st.write(f"Temperature T = {T:.4f}" + (" (default, uncalibrated)" if T == 1.0 else ""))

            except Exception as e:
                st.exception(e)
            finally:
                os.remove(tmp_path)

st.markdown("---")
st.caption("O5-T1/T2/T3 demo interface — for thesis evaluation purposes.")
