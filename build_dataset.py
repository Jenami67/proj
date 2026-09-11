"""End-to-end dataset build: read ASVspoof protocol files (train+dev+eval
combined) -> preprocess -> speaker-aware 60/20/20 split -> train-only
augmentation -> feature extraction -> save.

Run once: `python build_dataset.py`
Produces data/splits/{train,val,test}_{mfcc,mel,chroma,y}.npy
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


def build_features_for_split(paths, labels, split_name, augment):
    noise_files = (
        list(config.BACKGROUND_NOISE_DIR.glob("*.wav"))
        if config.BACKGROUND_NOISE_DIR.exists()
        else []
    )

    mfccs, mels, chromas, ys = [], [], [], []
    for i, (path, label) in enumerate(zip(paths, labels)):
        audio = audio_utils.preprocess_audio(path)
        if augment:
            audio = augmentation.apply_augmentations(audio, noise_files)

        feats = features.extract_all_features(audio)
        mfccs.append(feats["mfcc"])
        mels.append(feats["mel"])
        chromas.append(feats["chroma"])
        ys.append(label)

        if (i + 1) % 200 == 0 or (i + 1) == len(paths):
            print(f"[{split_name}] {i + 1}/{len(paths)} processed")

    return (
        np.array(mfccs, dtype=np.float32),
        np.array(mels, dtype=np.float32),
        np.array(chromas, dtype=np.float32),
        np.array(ys, dtype=np.int64),
    )


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
        mfcc, mel, chroma, y = build_features_for_split(
            split_paths, split_labels, split_name, augment
        )
        np.save(config.SPLITS_DIR / f"{split_name}_mfcc.npy", mfcc)
        np.save(config.SPLITS_DIR / f"{split_name}_mel.npy", mel)
        np.save(config.SPLITS_DIR / f"{split_name}_chroma.npy", chroma)
        np.save(config.SPLITS_DIR / f"{split_name}_y.npy", y)
        print(
            f"Saved {split_name}: {len(y)} samples "
            f"({int(np.sum(y == 0))} real, {int(np.sum(y == 1))} fake) "
            f"-> {config.SPLITS_DIR}"
        )


if __name__ == "__main__":
    main()
