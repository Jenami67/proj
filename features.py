"""Spectral feature extraction (Feature Extraction Layer, Fig. 1; FUN-03/04).

MFCC: 13 static coefficients + 13 delta + 13 delta-delta = 39-dim, matching
FUN-03's "39 coefficients" -- the standard convention for that number in
speech processing (static coefficients alone only give 13).

Mel-spectrogram: 128 mel bands, converted to log scale (dB), which is
standard practice -- raw linear power values are rarely fed to a CNN as-is.

Chroma: 12-dim chromagram (standard pitch-class representation).
"""

import numpy as np
import librosa

import config


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate([mfcc, delta, delta2], axis=0)  # (39, time)
    return combined.T  # (time, 39)


def extract_mel_spectrogram(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.SAMPLE_RATE, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # (time, 128)


def extract_chroma(audio):
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_chroma=config.N_CHROMA,
    )
    return chroma.T  # (time, 12)


def extract_all_features(audio):
    return {
        "mfcc": extract_mfcc(audio).astype(np.float32),
        "mel": extract_mel_spectrogram(audio).astype(np.float32),
        "chroma": extract_chroma(audio).astype(np.float32),
    }
