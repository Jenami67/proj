"""Central configuration for the deepfake voice detection pipeline."""

from pathlib import Path

# ---- Paths: official ASVspoof 2019 LA structure ----
# Point LA_ROOT at wherever you extracted the official "LA" folder.
# Expected layout (unmodified, straight from the official download):
#   LA/ASVspoof2019_LA_train/flac/*.flac
#   LA/ASVspoof2019_LA_dev/flac/*.flac
#   LA/ASVspoof2019_LA_eval/flac/*.flac
#   LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
#   LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
#   LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
PROJECT_DIR = Path(__file__).resolve().parent
LA_ROOT = PROJECT_DIR / "dataset" / "LA"

TRAIN_FLAC_DIR = LA_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_FLAC_DIR = LA_ROOT / "ASVspoof2019_LA_dev" / "flac"
EVAL_FLAC_DIR = LA_ROOT / "ASVspoof2019_LA_eval" / "flac"

PROTOCOLS_DIR = LA_ROOT / "ASVspoof2019_LA_cm_protocols"
TRAIN_PROTOCOL = PROTOCOLS_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOLS_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL = PROTOCOLS_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

# We combine train+dev+eval into one pool and then do our own speaker-independent
# 60/20/20 split (Section 3.1.4), rather than using ASVspoof's official partitions
# directly. This gives full control over the split ratios and is the standard
# approach for a project that isn't entering the ASVspoof challenge itself.
ALL_PARTITIONS = [
    (TRAIN_FLAC_DIR, TRAIN_PROTOCOL),
    (DEV_FLAC_DIR, DEV_PROTOCOL),
    (EVAL_FLAC_DIR, EVAL_PROTOCOL),
]

BACKGROUND_NOISE_DIR = PROJECT_DIR / "dataset" / "background_noise"  # optional

SPLITS_DIR = PROJECT_DIR / "data" / "splits"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

for _d in (SPLITS_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Audio (Table 3) ----
SAMPLE_RATE = 16000
DURATION_SECONDS = 4          # thesis says "4s if shorter, central 6s if longer";
                               # simplified here to always 4s -- see project notes.
N_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
PRE_EMPHASIS_ALPHA = 0.97
SILENCE_THRESHOLD_RATIO = 0.01  # 0.01 x peak amplitude, per Table 3

# ---- Features (FUN-03 / FUN-04) ----
N_MFCC = 13            # base coefficients; + delta + delta-delta = 39-dim total
N_MELS = 128
N_CHROMA = 12
N_FFT = 2048
HOP_LENGTH = 512

# ---- Split (Section 3.1.4) ----
RANDOM_STATE = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ---- Augmentation, training set only (Section 3.1.5, Step 3) ----
AUG_GAUSSIAN_NOISE_PROB = 0.30
AUG_BACKGROUND_NOISE_PROB = 0.20
AUG_SPEED_PERTURB_PROB = 0.30
AUG_PITCH_SHIFT_PROB = 0.20
AUG_MP3_COMPRESSION_PROB = 0.15

# ---- Training ----
# Batch size kept modest since the training hardware isn't confirmed yet
# (could end up being CPU-only). Safe to raise to 32/64 once you know your
# GPU's VRAM.
BATCH_SIZE = 16 # originally 2
EPOCHS = 3 # originally 50
LEARNING_RATE = 0.001
