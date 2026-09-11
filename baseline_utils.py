"""Shared helper for the classical ML baselines (XGBoost, AdaBoost).

These baselines don't use the raw time-series MFCC/Mel/chroma sequences the
deep learning models use -- classical ML models expect one fixed-length
flat vector per sample. The standard, simple way to get one from
time-series audio features is to summarize each feature dimension over
time using its mean and standard deviation. This is a simplified version
of the "temporal features" approach used by Bird & Lotfi (2023) for their
XGBoost baseline.
"""

import numpy as np

import config


def summarize_over_time(feature_array):
    """feature_array shape: (n_samples, time, n_feature_dims) ->
    returns (n_samples, n_feature_dims * 2): per-dimension mean and std."""
    mean = feature_array.mean(axis=1)
    std = feature_array.std(axis=1)
    return np.concatenate([mean, std], axis=1)


def load_flat_features(split_name):
    """Loads a split's saved features and flattens them into one vector
    per sample, combining MFCC + Mel + chroma."""
    mfcc = np.load(config.SPLITS_DIR / f"{split_name}_mfcc.npy")
    mel = np.load(config.SPLITS_DIR / f"{split_name}_mel.npy")
    chroma = np.load(config.SPLITS_DIR / f"{split_name}_chroma.npy")
    y = np.load(config.SPLITS_DIR / f"{split_name}_y.npy")

    X = np.concatenate(
        [
            summarize_over_time(mfcc),
            summarize_over_time(mel),
            summarize_over_time(chroma),
        ],
        axis=1,
    )
    return X, y
