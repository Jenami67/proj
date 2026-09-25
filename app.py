"""
app.py — Deepfake Voice Detection Interface (Objective 5)
Wired exactly to your config.py / features.py / model.py.

  O5-T1: accepts an audio file as input
  O5-T2: extracts features + runs the trained hybrid CNN+BiLSTM model
  O5-T3: displays real vs deepfake result
  FUN-07: shows a Grad-CAM explanation heatmap (CNN / Mel-spectrogram branch)

RUN:
    streamlit run app.py

ONE THING TO CONFIRM (marked TODO below): the saved model filename in
models/, and the bonafide/spoof label order used when you encoded labels
in build_dataset.py. Defaults assumed: 0 = bonafide, 1 = spoof.
"""

import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import glob

import config
from features import extract_mfcc, extract_mel_spectrogram
from gradcam_utils import compute_gradcam, overlay_heatmap_on_mel  # NEW

CLASS_NAMES = {0: "Bonafide (Real)", 1: "Spoof (Deepfake)"}  # TODO: confirm order matches build_dataset.py


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
            # Warm-up: run one dummy prediction now so TF's graph tracing
            # happens here (once, at startup) instead of during the user's
            # first click — this is usually what makes the first run feel
            # slow (~30s) while later runs are fast.
            dummy_frames = 1 + config.N_SAMPLES // config.HOP_LENGTH
            dummy_mel = np.zeros((1, dummy_frames, config.N_MELS), dtype=np.float32)
            dummy_mfcc = np.zeros((1, dummy_frames, config.N_MFCC * 3), dtype=np.float32)
            model([dummy_mel, dummy_mfcc], training=False)
            return model, str(path)

    raise FileNotFoundError(
        f"Couldn't find hybrid_cnn_rnn_best.keras or hybrid_cnn_rnn_final.keras "
        f"in {model_dir}. Files present: {[p.name for p in model_dir.glob('*.keras')]}"
    )


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


def predict(model, audio_path):
    y = preprocess_audio(audio_path)

    mel = extract_mel_spectrogram(y)   # (time, 128)
    mfcc = extract_mfcc(y)             # (time, 39)

    mel_batch = mel[np.newaxis, ...]    # (1, time, 128)
    mfcc_batch = mfcc[np.newaxis, ...]  # (1, time, 39)

    # Direct call (not .predict()) — much lower per-call overhead for a
    # single sample; .predict() builds a tf.data pipeline each time which
    # adds noticeable latency for one-off inference like this.
    probs = model([mel_batch, mfcc_batch], training=False).numpy()[0]  # softmax over 2 classes
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence, mel, mfcc  # CHANGED: also return features for Grad-CAM


def explain(model, mel, mfcc, predicted_class):
    """NEW: Grad-CAM heatmap for the predicted class, saved to a temp PNG.
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
                model, model_path = load_trained_model()
                predicted_class, confidence, mel, mfcc = predict(model, tmp_path)
                label = CLASS_NAMES[predicted_class]

                if predicted_class == 1:
                    st.error(f"### Result: {label}")
                else:
                    st.success(f"### Result: {label}")

                st.metric("Confidence", f"{confidence * 100:.1f}%")

                # NEW: explanation heatmap
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

            except Exception as e:
                st.exception(e)
            finally:
                os.remove(tmp_path)

st.markdown("---")
st.caption("O5-T1/T2/T3 demo interface — for thesis evaluation purposes.")
