"""Training-only data augmentation (Section 3.1.5, Step 3).

Each augmentation is applied independently and probabilistically per
sample, using the percentages given in the thesis as per-sample
probabilities (the more standard and simpler reading of "Gaussian noise
(30%)" etc. -- versus applying it to a fixed 30% subset of files).

The thesis calls this "online" augmentation, implying it happens fresh
every training epoch. This implementation applies it once, offline, when
the dataset is built, and caches the result to disk. This is a deliberate
simplification for a student-scale project: true epoch-by-epoch online
augmentation inside a tf.data pipeline is possible but adds real
complexity (librosa doesn't run natively inside TensorFlow's graph, so it
needs to be wrapped in tf.py_function). The training effect is very
similar either way -- the model still sees noisy/perturbed audio during
training -- it's just computed once instead of regenerated every epoch.
"""

import numpy as np
import librosa

import config
from audio_utils import standardize_duration, normalize_audio

rng = np.random.default_rng(config.RANDOM_STATE)


def add_gaussian_noise(audio, snr_db=15):
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(max(noise_power, 0)), size=audio.shape)
    return audio + noise


def add_background_noise(audio, noise_files, snr_db=10):
    """Mixes in a random background noise clip, if any were provided.
    Skips gracefully if no background noise files are available -- the
    thesis doesn't specify a noise source dataset, so this expects you to
    optionally drop .wav files into dataset/background_noise/."""
    if not noise_files:
        return audio
    noise_path = rng.choice(noise_files)
    noise, _ = librosa.load(noise_path, sr=config.SAMPLE_RATE, mono=True)
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    start = rng.integers(0, len(noise) - len(audio) + 1)
    noise = noise[start : start + len(audio)]

    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return audio
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return audio + noise


def speed_perturb(audio, rate_range=(0.9, 1.1)):
    rate = rng.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, semitone_range=(-2, 2)):
    n_steps = rng.uniform(*semitone_range)
    return librosa.effects.pitch_shift(audio, sr=config.SAMPLE_RATE, n_steps=n_steps)


def mp3_compression_emulation(audio, noise_floor_db=-40):
    """Approximates lossy-compression artifacts with quantization-style
    noise. A true MP3 encode/decode round-trip would need an external
    codec (pydub + ffmpeg); this is a lightweight standard substitute
    that doesn't require extra system dependencies."""
    noise_amp = 10 ** (noise_floor_db / 20)
    quant_noise = rng.uniform(-noise_amp, noise_amp, size=audio.shape)
    return audio + quant_noise


def apply_augmentations(audio, noise_files=None):
    """Applies each augmentation with its configured probability, then
    re-fixes duration and normalization (speed/pitch changes alter sample
    count and amplitude)."""
    if rng.random() < config.AUG_GAUSSIAN_NOISE_PROB:
        audio = add_gaussian_noise(audio)
    if rng.random() < config.AUG_BACKGROUND_NOISE_PROB:
        audio = add_background_noise(audio, noise_files or [])
    if rng.random() < config.AUG_SPEED_PERTURB_PROB:
        audio = speed_perturb(audio)
    if rng.random() < config.AUG_PITCH_SHIFT_PROB:
        audio = pitch_shift(audio)
    if rng.random() < config.AUG_MP3_COMPRESSION_PROB:
        audio = mp3_compression_emulation(audio)

    audio = standardize_duration(audio)
    audio = normalize_audio(audio)
    return audio
