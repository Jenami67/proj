"""One-off helper to build a TINY dummy ASVspoof-style dataset, just so
you can smoke-test the pipeline before the real dataset is downloaded.

This is a test utility, not part of the real pipeline -- delete it once
you've downloaded the actual ASVspoof2019 LA dataset. It does not touch
protocols.py, audio_utils.py, or build_dataset.py; it just arranges your
existing .wav files into the folder/protocol shape those files expect.

Design: each dummy "speaker" gets exactly one file labeled bonafide and
one labeled spoof. Since build_dataset.py's speaker-aware split moves
whole speaker groups together, this guarantees every split (train/val/
test) ends up with both classes, no matter how the random group split
falls -- not just "probably", but structurally guaranteed. Earlier
versions of this script assigned labels somewhat arbitrarily per file,
which could (and did) produce speakers with only one label, and from
there a test split with only one class in it.

Usage:
    python prepare_test_data.py /path/to/your/wav/files
"""

import shutil
import sys
from pathlib import Path

import config

MAX_FILES = 63  # uses however many .wav files are available, up to this cap


def main():
    if len(sys.argv) != 2:
        print("Usage: python prepare_test_data.py /path/to/your/wav/files")
        return

    source_dir = Path(sys.argv[1])
    wav_files = sorted(source_dir.glob("*.wav"))[:MAX_FILES]

    # Need an even count -- each dummy speaker consumes exactly 2 files
    # (one bonafide, one spoof). Drop the odd one out if necessary.
    if len(wav_files) % 2 != 0:
        wav_files = wav_files[:-1]

    if len(wav_files) < 4:
        print(
            f"Only found {len(wav_files)} usable .wav files in {source_dir}; "
            f"need at least 4 (2 dummy speakers)."
        )
        return

    n_speakers = len(wav_files) // 2
    print(
        f"Using {len(wav_files)} files -> {n_speakers} dummy speakers "
        f"(1 bonafide + 1 spoof file each)."
    )

    # Build one (speaker_id, flac_filename, label, source_wav) entry per
    # file, two consecutive entries per speaker: first = bonafide (real),
    # second = spoof (fake).
    entries = []
    for s in range(n_speakers):
        speaker_id = f"LA_TEST{s:03d}"
        real_wav, fake_wav = wav_files[2 * s], wav_files[2 * s + 1]
        entries.append((speaker_id, f"LA_T_TEST{s:03d}R", "bonafide", real_wav))
        entries.append((speaker_id, f"LA_T_TEST{s:03d}F", "spoof", fake_wav))

    # Wipe any previous test structure so this is repeatable.
    if config.LA_ROOT.exists():
        shutil.rmtree(config.LA_ROOT)

    # Spread speaker-pairs roughly 50/25/25 across train/dev/eval folders.
    # This particular split doesn't affect correctness -- build_dataset.py
    # pools train+dev+eval back into one list and does its own
    # speaker-independent 60/20/20 split regardless of how they're
    # arranged here. This just needs to be *some* valid partitioning so
    # the on-disk structure matches what protocols.py expects to read.
    half = n_speakers // 2
    three_quarters = (n_speakers * 3) // 4
    partitions = [
        (config.TRAIN_FLAC_DIR, config.TRAIN_PROTOCOL, range(0, half)),
        (config.DEV_FLAC_DIR, config.DEV_PROTOCOL, range(half, three_quarters)),
        (config.EVAL_FLAC_DIR, config.EVAL_PROTOCOL, range(three_quarters, n_speakers)),
    ]

    for flac_dir, protocol_path, speaker_range in partitions:
        flac_dir.mkdir(parents=True, exist_ok=True)
        protocol_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for s in speaker_range:
            for speaker_id, flac_name, label, wav_path in entries[2 * s : 2 * s + 2]:
                dest = flac_dir / f"{flac_name}.flac"
                # Copying the file with a .flac name -- the content is
                # still actual WAV data. soundfile/librosa read files by
                # inspecting the header, not the extension, so this
                # normally still loads fine for a smoke test. If it
                # errors out, tell me and we'll adjust.
                shutil.copy(wav_path, dest)
                system_id = "-" if label == "bonafide" else "A01"
                lines.append(f"{speaker_id} {flac_name} - {system_id} {label}")

        protocol_path.write_text("\n".join(lines) + "\n" if lines else "")
        print(f"Wrote {len(lines)} entries to {protocol_path}")

    print("\nDummy test dataset ready under:", config.LA_ROOT)
    print(
        f"\n{n_speakers} dummy speakers, each with exactly one bonafide and "
        "one spoof file. Every speaker group now contains both classes, so "
        "the speaker-aware split in build_dataset.py is guaranteed to "
        "produce both classes in train, val, AND test -- the 'only one "
        "class in y_true' warning should not happen again with this data. "
        "This still isn't a stand-in for real results (the audio content "
        "is arbitrary, not actual synthetic speech) -- it's purely a "
        "plumbing smoke test."
    )
    print("You can now run build_dataset.py as normal.")


if __name__ == "__main__":
    main()
