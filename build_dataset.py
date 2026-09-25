"""End-to-end dataset build: read ASVspoof protocol files (train+dev+eval
combined) -> preprocess -> speaker-aware 60/20/20 split -> train-only
augmentation -> feature extraction -> save.

Run once: `python build_dataset.py`
Produces data/splits/{train,val,test}_{mfcc,mel,chroma,y}.npy

Feature extraction streams straight to disk instead of accumulating every
sample in a Python list first. The original version appended each
sample's MFCC/Mel/chroma arrays to a list across the whole split, then
called np.array(...) once at the end -- for the ~67k-file train split,
that meant holding the entire split in RAM as loose per-sample arrays
AND, momentarily, a second full copy while NumPy allocated the final
contiguous array to stack them into. On Colab's free tier (limited RAM),
that peak moment was enough to get the process OOM-killed silently --
no Python traceback, since the kernel kills the process directly rather
than raising a catchable exception. This version pre-allocates the
on-disk .npy file at its final size up front (via
np.lib.format.open_memmap) and writes each sample directly into its
slot as soon as it's computed, so peak memory stays around one sample's
worth, not the whole split's.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import config
import audio_utils
import augmentation
import features
import protocols


def gather_files():
    """Combines train+dev+eval into one pool, reading labels and speaker
    IDs straight from the official protocol files.
    Label convention: 0 = real (bonafide), 1 = fake (spoof)."""
    all_paths, all_labels, all_speakers = [], [], []

    for flac_dir, protocol_path in config.ALL_PARTITIONS:
        paths, labels, speakers = protocols.build_file_list(flac_dir, protocol_path)
        all_paths.extend(paths)
        all_labels.extend(labels)
        all_speakers.extend(speakers)
        print(f"  {protocol_path.name}: {len(paths)} files")

    return (
        np.array(all_paths, dtype=object),
        np.array(all_labels),
        np.array(all_speakers),
    )


def speaker_aware_split(paths, labels, speakers):
    """60/20/20 split where no speaker appears in more than one subset
    (Section 3.1.4)."""
    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=config.TRAIN_RATIO, random_state=config.RANDOM_STATE
    )
    train_idx, temp_idx = next(gss1.split(paths, labels, groups=speakers))

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=config.RANDOM_STATE)
    val_rel, test_rel = next(
        gss2.split(paths[temp_idx], labels[temp_idx], groups=speakers[temp_idx])
    )
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]


def probe_feature_shapes(sample_path, augment, noise_files):
    """Runs the full preprocessing+feature pipeline on one file to learn
    the exact output shapes before allocating the on-disk arrays. All
    clips are padded/cropped to the same duration by
    audio_utils.standardize_duration(), so every sample in a split
    produces the same (time, n_features) shape -- that constant shape is
    what makes pre-allocation possible."""
    audio = audio_utils.preprocess_audio(sample_path)
    if augment:
        audio = augmentation.apply_augmentations(audio, noise_files)
    feats = features.extract_all_features(audio)
    return {
        "mfcc": feats["mfcc"].shape,
        "mel": feats["mel"].shape,
        "chroma": feats["chroma"].shape,
    }


def build_features_for_split(paths, labels, split_name, augment):
    """Streams features straight to pre-allocated on-disk .npy files
    instead of building Python lists in memory. Returns the label counts
    for the summary print, not the feature arrays themselves -- callers
    that need the data back should np.load() it from disk, the same way
    every other script in this project already does."""
    noise_files = (
        list(config.BACKGROUND_NOISE_DIR.glob("*.wav"))
        if config.BACKGROUND_NOISE_DIR.exists()
        else []
    )

    n_samples = len(paths)
    if n_samples == 0:
        # Still create empty files so downstream scripts that np.load()
        # every split don't hit a missing-file error.
        for feat_name in ("mfcc", "mel", "chroma"):
            np.save(config.SPLITS_DIR / f"{split_name}_{feat_name}.npy", np.array([]))
        np.save(config.SPLITS_DIR / f"{split_name}_y.npy", np.array([], dtype=np.int64))
        return 0, 0

    # Probe the first file to learn output shapes, then pre-allocate the
    # full on-disk arrays at their final size. This is the key change:
    # NumPy never needs to hold the whole split's worth of samples in
    # RAM at once to build these arrays -- each array already exists on
    # disk at full size, and we just write into it slice by slice below.
    shapes = probe_feature_shapes(paths[0], augment, noise_files)

    mfcc_arr = np.lib.format.open_memmap(
        config.SPLITS_DIR / f"{split_name}_mfcc.npy", mode="w+",
        dtype=np.float32, shape=(n_samples, *shapes["mfcc"]),
    )
    mel_arr = np.lib.format.open_memmap(
        config.SPLITS_DIR / f"{split_name}_mel.npy", mode="w+",
        dtype=np.float32, shape=(n_samples, *shapes["mel"]),
    )
    chroma_arr = np.lib.format.open_memmap(
        config.SPLITS_DIR / f"{split_name}_chroma.npy", mode="w+",
        dtype=np.float32, shape=(n_samples, *shapes["chroma"]),
    )
    y_arr = np.lib.format.open_memmap(
        config.SPLITS_DIR / f"{split_name}_y.npy", mode="w+",
        dtype=np.int64, shape=(n_samples,),
    )

    for i, (path, label) in enumerate(zip(paths, labels)):
        audio = audio_utils.preprocess_audio(path)
        if augment:
            audio = augmentation.apply_augmentations(audio, noise_files)

        feats = features.extract_all_features(audio)

        # Written directly into the on-disk array's slot -- no list
        # accumulation, so peak memory stays around one sample's worth
        # (plus small OS-level write buffering), regardless of how many
        # files are in the split.
        mfcc_arr[i] = feats["mfcc"]
        mel_arr[i] = feats["mel"]
        chroma_arr[i] = feats["chroma"]
        y_arr[i] = label

        if (i + 1) % 200 == 0 or (i + 1) == n_samples:
            print(f"[{split_name}] {i + 1}/{n_samples} processed")
            # Periodically flush to disk so a crash mid-split loses at
            # most the last ~200 samples' progress, not everything --
            # flush() forces the memmap's OS write buffer out rather
            # than trusting it to happen eventually on its own.
            mfcc_arr.flush()
            mel_arr.flush()
            chroma_arr.flush()
            y_arr.flush()

    n_real = int(np.sum(y_arr == 0))
    n_fake = int(np.sum(y_arr == 1))

    # Dropping the memmap references (and closing over the loop) lets
    # the OS release the file handles cleanly.
    del mfcc_arr, mel_arr, chroma_arr, y_arr

    return n_real, n_fake


def main():
    print("Reading protocol files...")
    paths, labels, speakers = gather_files()
    print(
        f"\nFound {len(paths)} files total "
        f"({int(np.sum(labels == 0))} real, {int(np.sum(labels == 1))} fake), "
        f"{len(np.unique(speakers))} unique speakers."
    )
    if len(paths) == 0:
        print(
            "No files found. Check that config.LA_ROOT points at your "
            "extracted ASVspoof2019 LA folder."
        )
        return

    train_idx, val_idx, test_idx = speaker_aware_split(paths, labels, speakers)

    # Sanity check: confirm no speaker overlap across splits.
    train_speakers = set(speakers[train_idx])
    val_speakers = set(speakers[val_idx])
    test_speakers = set(speakers[test_idx])
    overlap = (train_speakers & val_speakers) | (train_speakers & test_speakers) | (val_speakers & test_speakers)
    assert not overlap, f"Speaker leakage detected across splits: {overlap}"
    print("Speaker-independence check passed: no speaker appears in more than one split.")

    # Augmentation is applied to the training split only (Section 3.1.5).
    splits = {
        "train": (paths[train_idx], labels[train_idx], True),
        "val": (paths[val_idx], labels[val_idx], False),
        "test": (paths[test_idx], labels[test_idx], False),
    }

    for split_name, (split_paths, split_labels, augment) in splits.items():
        n_real, n_fake = build_features_for_split(
            split_paths, split_labels, split_name, augment
        )
        print(
            f"Saved {split_name}: {len(split_labels)} samples "
            f"({n_real} real, {n_fake} fake) -> {config.SPLITS_DIR}"
        )


if __name__ == "__main__":
    main()

