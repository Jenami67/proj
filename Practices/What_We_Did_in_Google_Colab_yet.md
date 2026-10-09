# Deepfake Voice Detection Project

## Google Colab Workflow – From Zero to Training

This document explains, in simple terms, **how the project is run in Google Colab, up to and including the training phase**.

The previous document explains what each part of the AI system does. This one explains:

> **Which commands do we run, in which order, to get from an empty Colab session to a trained model?**

The project is a **hybrid CNN + BiLSTM** detector (Mel-spectrogram → CNN branch, MFCC sequence → BiLSTM branch, fused before classification), trained on **ASVspoof 2019 LA**.

The general idea:

**Open Colab → connect Google Drive → copy code → extract dataset → build features → back them up → test the pipeline → train**

Every step below has a matching cell in the notebook `deepfake_detector_colab.ipynb`.

---

# 1. Start Google Colab

Colab gives us a temporary Linux computer in the cloud. Training on a normal computer would take much longer.

Before running anything, switch to a GPU:

**Runtime → Change runtime type → T4 GPU**

Colab's local storage (`/content`) disappears when the session ends, so anything important must be saved to **Google Drive**.

---

# 2. Connect Google Drive and Define the Paths

```python
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/deepfake_detector"   # Drive folder that contains "code/"
ZIP  = f"{BASE}/dataset/zip_file/LA.zip"            # ASVspoof2019 LA zip on Drive
WORK = "/content/project"                           # fast local working copy (temporary)
```

Our Drive folder looks like this:

```text
MyDrive/deepfake_detector
├── code/                  ← all the .py files (including audio_utils.py)
├── dataset/zip_file/LA.zip
├── models/                ← created automatically, holds checkpoints
├── results/               ← created automatically
└── data/splits_backup/    ← feature backup (step 11)
```

The idea:

> Google Drive = safe storage
> `/content/project` = temporary but fast working space

---

# 3. Check That the GPU Is Available

```bash
!nvidia-smi | head -n 12
!python -c "import tensorflow as tf; print('GPUs visible:', tf.config.list_physical_devices('GPU'))"
```

We run this in a separate process on purpose, so the notebook itself never grabs GPU memory that training will need later.

---

# 4. Copy the Code to Local Disk and Link `models/` and `results/` to Drive

The code runs from `/content/project` (fast), but trained models and results must survive a disconnect. So we copy the code locally and make `models/` and `results/` **symbolic links** into Drive:

```python
import os

os.makedirs(WORK, exist_ok=True)
!cp -f "{BASE}"/code/*.py "{WORK}"/
%cd {WORK}

for name in ("models", "results"):
    os.makedirs(f"{BASE}/{name}", exist_ok=True)
    link = f"{WORK}/{name}"
    if os.path.islink(link):
        os.remove(link)
    elif os.path.isdir(link):
        os.rmdir(link)          # only succeeds if empty
    os.symlink(f"{BASE}/{name}", link)
```

In simple terms:

> "When the program writes into `models/`, secretly save it into the Drive folder."

The code files on Drive are the master copy. If you edit a file, edit it in `{BASE}/code` and run the `cp` line again.

---

# 5. Install Required Python Packages

```bash
!pip install -q librosa soundfile
```

TensorFlow, NumPy and scikit-learn are already available in Colab. Nothing else is needed: the XGBoost/AdaBoost baselines and Grad-CAM tooling are no longer part of the project.

---

# 6. Extract the ASVspoof Dataset to Local Disk

Reading 121,461 small audio files straight from Drive is very slow, and unzipping straight from Drive is slow too. So we copy the zip to Colab's local disk first, unzip there, and delete the local zip:

```bash
!cp "{ZIP}" /content/LA.zip
!mkdir -p "{WORK}/dataset"
!unzip -q -o /content/LA.zip -d "{WORK}/dataset"
!rm -f /content/LA.zip
```

`unzip -q` prints nothing while it works, so be patient. The notebook cell also writes a small marker file when extraction finishes, so re-running the cell doesn't extract twice.

---

# 7. Verify the Dataset Layout

`config.py` expects the data at exactly:

```text
dataset/LA/ASVspoof2019_LA_train/flac/*.flac
dataset/LA/ASVspoof2019_LA_dev/flac/*.flac
dataset/LA/ASVspoof2019_LA_eval/flac/*.flac
dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

The notebook imports `config` and prints OK / MISSING for each path, plus the number of flac files. Expected counts:

| Partition | Files  |
| --------- | ------ |
| train     | 25,380 |
| dev       | 24,844 |
| eval      | 71,237 |
| **Total** | **121,461** |

The folders `ASVspoof2019_LA_asv_protocols` and `ASVspoof2019_LA_asv_scores` are for speaker verification and are not used by this project.

If the folders are nested one level too deep (for example `dataset/LA/LA/...`), fix it with:

```bash
!mv dataset/LA/LA/* dataset/LA/
```

and re-run the check.

---

# 8. Build the Features

```bash
!df -h /content | tail -n 1
!python build_dataset.py
```

This one command runs the whole preparation pipeline:

**Audio files → preprocessing → speaker-independent 60/20/20 split → augmentation (train only) → MFCC + Mel features → saved `.npy` files**

What to expect:

* It first prints the file counts and **"Speaker-independence check passed"**. If it says "No files found", the layout from step 7 is wrong.
* It then processes train, val and test, printing progress every 200 files.
* It runs on the CPU and takes a long time. Keep the tab open.
* Features are streamed straight to disk, so RAM stays low. If the session dies in the middle of a split, that split restarts from scratch.

Only two feature types are produced now (chroma features were removed):

```text
data/splits/
├── train_mfcc.npy   train_mel.npy   train_y.npy
├── val_mfcc.npy     val_mel.npy     val_y.npy
└── test_mfcc.npy    test_mel.npy    test_y.npy
```

Check that they exist:

```bash
!ls -lh data/splits/
```

In simple terms:

> `build_dataset.py` does the heavy preparation work.
> The `.npy` files are the prepared food the model will eat.

---

# 9. Back Up the Features to Google Drive

Because `/content` is temporary, copy the generated features to Drive:

```bash
!du -sh data/splits
!mkdir -p "{BASE}/data/splits_backup"
!cp data/splits/*.npy "{BASE}/data/splits_backup/"
!ls -lh "{BASE}/data/splits_backup/"
```

Check the `du` size against your free Drive space first. The free tier has 15 GB, and the dataset zip already uses a good part of it. If Drive is too full, skip the backup, but then a disconnect means rebuilding. (The notebook never reads the already-extracted `zip_file/LA` folder on Drive, so if you have one, you can delete it to free space.)

---

# 10. Restore the Features in a New Colab Session

If the runtime restarts, `/content` is wiped. Instead of rebuilding 121,461 files, restore the backup:

1. Re-run the steps 2, 4 and 5 (mount Drive, copy code and links, install packages).
2. Copy the features back:

```bash
!mkdir -p data/splits
!cp "{BASE}/data/splits_backup/"*.npy data/splits/
!ls -lh data/splits/
```

You do **not** need to extract the dataset again for training. Only `build_dataset.py` needs the audio files.

---

# 11. Quick Pipeline Test (One Batch)

Before the long training run, check that the training code can actually read the data. This runs `load_split` and `make_dataset` from `train.py` and pulls a single batch:

```python
import tensorflow as tf
from train import load_split, make_dataset

mel, mfcc, y = load_split("train")
y_cat = tf.keras.utils.to_categorical(y, num_classes=2)
ds = make_dataset(mel, mfcc, y_cat, batch_size=32, shuffle=True)

for (mel_b, mfcc_b), y_b in ds.take(1):
    print("mel batch:", mel_b.shape, "| mfcc batch:", mfcc_b.shape, "| y batch:", y_b.shape)
print("Generator pipeline test passed.")
```

If it prints `Generator pipeline test passed.`, then the features load, labels are available, and the TensorFlow data pipeline works. The notebook runs this in a separate process (like the GPU check) so training starts with a free GPU.

---

# 12. Train the Hybrid Model

```bash
!python train.py
```

This trains the **hybrid CNN + BiLSTM** model only. It handles:

* Loading the prepared data lazily from disk
* Creating batches (batch size and epochs come from `config.py`: 32 and 50)
* Class weighting
* Validation after every epoch
* Early stopping (patience 8 on validation loss)
* Saving the best checkpoint and the epoch counter after every epoch

Files it writes, which land in your Drive `models/` folder through the link from step 4:

```text
models/hybrid_cnn_rnn_best.keras          ← best checkpoint (lowest val loss)
models/hybrid_training_progress.json      ← completed-epoch counter
models/hybrid_cnn_rnn_final.keras         ← saved when training ends
```

Check them with:

```bash
!ls -lh models/
```

---

# 13. If Training Is Interrupted

Free Colab sessions can disconnect or lose the GPU. Because checkpoints go to Drive, nothing is lost:

> **Train → save progress → Colab stops → reconnect → continue**

To continue:

1. Re-run steps 2, 4 and 5.
2. If `data/splits` is empty, restore it (step 10).
3. Run:

```bash
!python train.py --resume
```

This loads `hybrid_cnn_rnn_best.keras` and continues from the epoch recorded in the progress file. Note that the best checkpoint is the lowest-validation-loss one, which may be a few epochs behind the last epoch that ran.

---

# 14. The Whole Process in One Picture

```text
1. Open Colab, switch to GPU
        ↓
2. Mount Google Drive, set BASE / ZIP / WORK
        ↓
3. Check GPU
        ↓
4. Copy code to /content/project, link models/ and results/ to Drive
        ↓
5. Install librosa + soundfile
        ↓
6. Extract LA.zip to /content/project/dataset
        ↓
7. Verify the dataset layout
        ↓
8. Run build_dataset.py
        ↓
9. Back up data/splits to Drive
        ↓
10. (new session only) Restore data/splits from Drive
        ↓
11. One-batch pipeline test
        ↓
12. Run train.py
        ↓
13. (if interrupted) Run train.py --resume
```

---

# 15. The Commands We Mainly Use

```bash
!pip install -q librosa soundfile                         # packages
!cp "{ZIP}" /content/LA.zip                               # copy zip locally
!unzip -q -o /content/LA.zip -d "{WORK}/dataset"          # extract dataset
!python build_dataset.py                                  # build features
!ls -lh data/splits/                                      # check features
!cp data/splits/*.npy "{BASE}/data/splits_backup/"        # back up to Drive
!python train.py                                          # train
!python train.py --resume                                 # continue after a disconnect
!ls -lh models/                                           # check saved models
```

---

# 16. What Each Python File Does

| File                   | Job                                                                 |
| ---------------------- | ------------------------------------------------------------------- |
| `config.py`            | Store paths, audio settings, split ratios, augmentation, training settings |
| `protocols.py`         | Read the ASVspoof protocol files (labels and speaker IDs)           |
| `audio_utils.py`       | Load and preprocess the audio (settings come from `config.py`)      |
| `augmentation.py`      | Training-only augmentation (noise, speed, pitch, MP3-style)         |
| `features.py`          | Extract the 39-dim MFCC and the log-Mel spectrogram                 |
| `build_dataset.py`     | Run the whole pipeline: split, augment, extract features, save `.npy` |
| `model.py`             | Define the hybrid CNN + BiLSTM architecture                         |
| `train.py`             | Train the hybrid model, with checkpoints and `--resume`             |
| `metrics_utils.py`     | Metric calculations (used later by evaluation)                      |
| `evaluate.py`          | Test the trained model on the test split (next phase, not covered here) |
| `prepare_test_data.py` | Optional: build a tiny dummy dataset for a smoke test               |

> ⚠️ **`prepare_test_data.py` deletes the whole `dataset/LA` folder** before building its dummy data. Never run it once the real dataset is in place. Delete it whenever you like.

So instead of thinking "there are many complicated Python files", think:

> **Each file has one main job.**

---

# 17. The Most Important Thing to Understand

We do **not** manually type hundreds of commands to process every audio file. The Python programs are written once, and Colab runs them.

```bash
!python build_dataset.py
```

means: *"Hey Python, run the program called `build_dataset.py`."* It then processes 121,461 audio files automatically.

```bash
!python train.py
```

means: *"Run the training program."*

---

# 18. The Simplest Possible Explanation

If you remember only one thing, remember this:

### Step 1 – Prepare

```bash
!python build_dataset.py
```

**Turn the huge collection of audio files into prepared training data.**

### Step 2 – Train

```bash
!python train.py
```

**Teach the hybrid model using that prepared data.**

So up to this point, the whole project reduces to:

> **Prepare → Train**

Evaluation (`evaluate.py`) comes next, but is not covered in this document.
