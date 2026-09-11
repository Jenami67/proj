"""Fixed audio preprocessing chain, following the order given in Table 3:
resample -> silence trim -> pre-emphasis -> peak normalize -> duration fix.

Note: normalization runs LAST, after pre-emphasis. Pre-emphasis changes the
signal's amplitude scale, so normalizing before it (as in an earlier version
of this pipeline) doesn't actually guarantee the final [-1, 1] range the
thesis asks for. Running it last does.
"""

import numpy as np
import librosa

import config


def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    return audio


def trim_silence(audio):
    """Approximate energy-based VAD trimming using an amplitude threshold.

    The thesis specifies "Voice Activity Detection (energy threshold =
    0.01 x max amplitude)". librosa has no direct energy-threshold VAD
    function, so this converts that ratio into an equivalent top_db value
    for librosa.effects.trim(), which is the standard, closest available
    tool for this in librosa.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    top_db = -20 * np.log10(config.SILENCE_THRESHOLD_RATIO)
    trimmed, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=2048, hop_length=config.HOP_LENGTH
    )
    return trimmed


def pre_emphasis(audio, alpha=config.PRE_EMPHASIS_ALPHA):
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])


def normalize_audio(audio):
    """Peak normalization to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def standardize_duration(audio, n_samples=config.N_SAMPLES):
    """Pads with zeros or center-crops so every clip has identical length --
    required so samples can be stacked into a uniform-shape array/tensor."""
    length = len(audio)
    if length < n_samples:
        audio = np.pad(audio, (0, n_samples - length))
    elif length > n_samples:
        start = (length - n_samples) // 2
        audio = audio[start : start + n_samples]
    return audio


def preprocess_audio(file_path):
    """Runs the full fixed chain on one file, in Table 3's order."""
    audio = load_audio(file_path)
    audio = trim_silence(audio)
    audio = pre_emphasis(audio)
    audio = normalize_audio(audio)
    audio = standardize_duration(audio)
    return audio
