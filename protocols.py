"""Parses ASVspoof2019 LA CM protocol files.

Each line looks like:
    LA_0079 LA_T_1138215 - - bonafide
    LA_0079 LA_T_1271820 - A01 spoof

Columns: SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY
KEY is "bonafide" or "spoof" -- this is where the ground-truth label
actually comes from (not the filename, not the folder).
"""

from pathlib import Path


def parse_protocol(protocol_path):
    """Returns a list of (filename_stem, speaker_id, label) tuples.
    label: 0 = real (bonafide), 1 = fake (spoof)."""
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip malformed/blank lines
            speaker_id, filename, _unused, system_id, key = parts
            label = 0 if key == "bonafide" else 1
            entries.append((filename, speaker_id, label))
    return entries


def build_file_list(flac_dir, protocol_path, extension=".flac"):
    """Combines a protocol file with its matching audio folder.
    Returns (paths, labels, speaker_ids) as parallel lists, skipping any
    protocol entries whose audio file isn't actually present on disk."""
    flac_dir = Path(flac_dir)
    entries = parse_protocol(protocol_path)

    paths, labels, speakers = [], [], []
    missing = 0
    for filename, speaker_id, label in entries:
        file_path = flac_dir / f"{filename}{extension}"
        if not file_path.exists():
            missing += 1
            continue
        paths.append(file_path)
        labels.append(label)
        speakers.append(speaker_id)

    if missing:
        print(
            f"Note: {missing} file(s) listed in {protocol_path.name} were not "
            f"found in {flac_dir} and were skipped."
        )
    return paths, labels, speakers
