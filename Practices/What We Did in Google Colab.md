# Deepfake Voice Detection Project

## What We Did in Google Colab – Step by Step

This document explains, in simple terms, **how the project was actually executed in Google Colab**.

The previous document explains what each part of the AI system does.

This document explains:

> **What did we actually do and which commands did we run to make it happen?**

The general idea was:

**Open Colab → connect Google Drive → prepare dataset → build dataset → train models → evaluate models → run comparisons → generate Grad-CAM**

---

# 1. Start Google Colab

The project was run in **Google Colab**.

Colab gives us a temporary Linux computer in the cloud. We used it because training the models on a normal computer would take much longer and would require more computing resources.

However, Colab's temporary storage can disappear when the session ends.

Therefore, the important project files were kept in:

**Google Drive**

---

# 2. Connect Google Drive

The first important step was connecting Colab to Google Drive.

We ran:

```python
from google.colab import drive

drive.mount('/content/drive')
```

Colab then asked us to authorize access to Google Drive.

After connecting, we defined the project location:

```python
PROJECT_DIR = '/content/drive/MyDrive/deepfake_detector'
```

So our main project folder was:

```text
Google Drive
└── MyDrive
    └── deepfake_detector
```

This folder contained the project code, dataset files, models, results and other important files.

---

# 3. Check That the GPU Is Available

Before doing expensive AI work, we checked whether Colab had given us a GPU.

We ran:

```python
import tensorflow as tf

print("GPUs visible:", tf.config.list_physical_devices('GPU'))
```

If TensorFlow showed a GPU, the environment could use it for neural-network training.

This was especially useful because we were using the free Colab environment, where GPU availability is not always guaranteed.

---

# 4. Open the Project Folder

We then worked inside the project directory.

The important files were checked with:

```bash
!ls
```

This allowed us to make sure files such as:

```text
build_dataset.py
train.py
evaluate.py
xgboost_baseline.py
adaboost_baseline.py
gradcam_utils.py
config.py
```

were available.

In some sessions we used:

```python
%cd /content/drive/MyDrive/deepfake_detector
```

to move into the project folder.

---

# 5. Prepare the ASVspoof Dataset

The ASVspoof dataset was stored as:

```text
dataset/LA.zip
```

Because Colab's local storage is temporary, we extracted the ZIP file into Colab's local storage:

```text
/content/dataset
```

The important command was:

```bash
!unzip -q /content/drive/MyDrive/deepfake_detector/dataset/LA.zip -d /content/dataset
```

After extraction, we checked that the dataset existed:

```bash
!ls /content/dataset/LA
```

The reason for extracting it to `/content` instead of processing everything directly from Google Drive was mainly **speed**.

In simple terms:

> Google Drive = safe storage
> `/content` = temporary but faster working space

---

# 6. Make the Dataset Accessible to the Project

The project code expected the dataset at:

```text
project/dataset/LA
```

but the actual extracted dataset was at:

```text
/content/dataset/LA
```

So we connected the two locations using a symbolic link.

In simple terms:

> "When the program asks for `dataset/LA`, secretly send it to `/content/dataset/LA`."

This allowed the existing project code to work without changing all of its paths.

---

# 7. Install Required Python Packages

Before running the project, we installed the packages needed by the code.

For example:

```bash
!pip install -q xgboost librosa soundfile
```

These provide functionality for things such as:

* Audio processing
* Feature extraction
* XGBoost training
* Reading and writing audio files

TensorFlow and some other packages were already available in the Colab environment.

---

# 8. Build the Dataset

This was the main command for creating the processed dataset:

```bash
!python build_dataset.py
```

This single command started the dataset-building pipeline.

The script handled the work described in the previous article, including:

**Audio files → preprocessing → augmentation → feature extraction → saved feature data**

The script read the ASVspoof protocol files to determine which recordings were:

* Real
* Fake

It then processed the audio and created the data needed for training.

---

# 9. Check the Generated Dataset

After the dataset-building process finished, we checked whether the expected files had been created.

We ran:

```bash
!ls -lh /content/data/splits/
```

This showed the generated `.npy` files.

These files contained the processed data used later by the training program.

In simple terms:

> `build_dataset.py` did the heavy preparation work.
> The `.npy` files were the prepared food that the training model could eat.

---

# 10. Save a Backup of the Generated Data

Because `/content` is temporary, we copied the generated split files to Google Drive.

We created a backup folder:

```python
import os

os.makedirs(
    '/content/drive/MyDrive/deepfake_detector/data/splits_backup',
    exist_ok=True
)
```

Then copied the generated files:

```bash
!cp -r /content/data/splits/*.npy /content/drive/MyDrive/deepfake_detector/data/splits_backup/
```

We checked the backup:

```bash
!ls -lh /content/drive/MyDrive/deepfake_detector/data/splits_backup/
```

This was important because if Colab's runtime disappeared, the generated dataset would not have to be rebuilt from the beginning.

---

# 11. Restore the Dataset in a New Colab Session

Sometimes the Colab runtime was restarted or disconnected.

The temporary `/content` storage could then disappear.

Instead of running the entire dataset-building process again, we restored the saved split files from Google Drive.

We checked:

```bash
!ls -lh data/splits_backup/
```

Then the backup was moved back into the location expected by the project:

```bash
!mv data/splits_backup data/splits
```

After that:

```bash
!ls -lh data/splits/
```

confirmed that the training data was available again.

The important idea is:

> **We did not want to rebuild 121,461 audio files every time Colab restarted.**

---

# 12. Check the Training Data Pipeline

Before starting a long training session, we performed a small test to make sure the training program could actually read the data.

We loaded the training split:

```python
train_mel, train_mfcc, y_train = load_split("train")
```

The labels were converted to the format expected by the neural networks:

```python
y_train_cat = tf.keras.utils.to_categorical(
    y_train,
    num_classes=2
)
```

Then we created the TensorFlow dataset:

```python
ds = make_dataset(
    train_mel,
    train_mfcc,
    y_train_cat,
    batch_size=32,
    shuffle=True
)
```

We took only one batch:

```python
for (mel_batch, mfcc_batch), y_batch in ds.take(1):
    print("mel batch shape:", mel_batch.shape)
    print("mfcc batch shape:", mfcc_batch.shape)
    print("y batch shape:", y_batch.shape)
```

If this worked, we knew that:

* The files could be loaded.
* Mel features were available.
* MFCC features were available.
* Labels were available.
* The TensorFlow data pipeline was working.

We printed:

```text
Generator pipeline test passed.
```

This was simply a quick **"Is everything connected correctly?"** test before starting the expensive training.

---

# 13. Train the Neural Networks

The main training command was:

```bash
!python train.py
```

This was the main training program.

It trained the neural-network models according to the configuration in the project.

The three neural-network approaches were:

```text
CNN-only
RNN-only
Hybrid CNN-RNN
```

The training program handled things such as:

* Loading the prepared data
* Creating batches
* Training the neural network
* Validation
* Class weighting
* Early stopping
* Saving checkpoints
* Saving training progress

So we did **not** manually train each neural-network layer in Colab.

We simply started the training program:

```bash
!python train.py
```

and the Python code handled the training process.

---

# 14. If Training Was Interrupted

Training could take a long time.

Also, free Google Colab sessions can disconnect or lose GPU access.

Therefore, the project included checkpoint support.

The best model was saved during training.

If training was interrupted, we could continue using the saved checkpoint rather than starting completely from zero.

This was particularly important for this project because the dataset was large.

The basic idea was:

> **Train → save progress → Colab stops → reconnect → continue**

rather than:

> **Train → Colab stops → cry → start everything again**

---

# 15. Evaluate the Hybrid CNN-RNN

After training, we evaluated the hybrid model using:

```bash
!python evaluate.py --model hybrid
```

This loaded the trained hybrid model and tested it against the test set.

It produced the evaluation results containing measurements such as:

* Accuracy
* Precision
* Recall
* Specificity
* F1-score
* ROC-AUC
* EER
* Confusion matrix values
* Inference time

The results were saved to a JSON file.

---

# 16. Evaluate the CNN-only Model

We then tested the CNN-only model:

```bash
!python evaluate.py --model cnn_only
```

This allowed us to see how the CNN performed without the RNN branch.

Again, the evaluation results were saved.

---

# 17. Evaluate the RNN-only Model

We also tested the RNN-only model:

```bash
!python evaluate.py --model rnn_only
```

Now we had results for all three neural-network approaches:

```text
CNN-only
RNN-only
Hybrid CNN-RNN
```

This was necessary because the goal was not simply to build one model.

We wanted to compare:

> CNN vs RNN vs CNN+RNN

---

# 18. Check the Trained Models

After training, we checked the model directory:

```bash
!ls -lh models/
```

This allowed us to confirm that the trained model files had actually been created.

For example, the project produced files such as:

```text
hybrid_cnn_rnn_best.keras
hybrid_cnn_rnn_final.keras

cnn_only_best.keras
cnn_only_final.keras

rnn_only_best.keras
rnn_only_final.keras
```

---

# 19. Run the XGBoost Baseline

The XGBoost comparison model had its own Python script.

We ran:

```bash
!python xgboost_baseline.py
```

This trained and evaluated the XGBoost model using the prepared feature data.

The purpose was to have a traditional machine-learning model to compare against the neural networks.

---

# 20. Run the AdaBoost Baseline

We did the same for AdaBoost:

```bash
!python adaboost_baseline.py
```

This produced the AdaBoost comparison results.

At this point we had tested:

```text
CNN-only
RNN-only
Hybrid CNN-RNN
XGBoost
AdaBoost
```

These results were later collected into the comparison table in the previous article.

---

# 21. Generate Grad-CAM

After the models were trained, we also ran the Grad-CAM utility:

```bash
!python gradcam_utils.py
```

Grad-CAM generates visual information showing which areas of the Mel-spectrogram influenced the CNN-based prediction.

The purpose was to make the model more understandable.

Instead of only getting:

> "Fake"

we can also investigate:

> "Which part of the sound representation influenced this decision?"

---

# 22. The Whole Process in One Picture

The entire Colab workflow can be remembered like this:

```text
1. Open Google Colab
        ↓
2. Mount Google Drive
        ↓
3. Check GPU
        ↓
4. Open project folder
        ↓
5. Prepare / extract ASVspoof dataset
        ↓
6. Install required packages
        ↓
7. Run build_dataset.py
        ↓
8. Check generated split files
        ↓
9. Back up important data to Google Drive
        ↓
10. Check training data pipeline
        ↓
11. Run train.py
        ↓
12. Models are trained and checkpoints saved
        ↓
13. Evaluate Hybrid
        ↓
14. Evaluate CNN-only
        ↓
15. Evaluate RNN-only
        ↓
16. Run XGBoost
        ↓
17. Run AdaBoost
        ↓
18. Run Grad-CAM
        ↓
19. Results saved
```

---

# 23. The Commands We Mainly Used

If the monkey only wants to remember the **important commands**, these are the ones:

### Check files

```bash
!ls
```

### Install packages

```bash
!pip install -q xgboost librosa soundfile
```

### Extract the dataset

```bash
!unzip -q dataset/LA.zip -d /content/dataset
```

### Build the processed dataset

```bash
!python build_dataset.py
```

### Check generated data

```bash
!ls -lh /content/data/splits/
```

### Train neural networks

```bash
!python train.py
```

### Evaluate Hybrid CNN-RNN

```bash
!python evaluate.py --model hybrid
```

### Evaluate CNN-only

```bash
!python evaluate.py --model cnn_only
```

### Evaluate RNN-only

```bash
!python evaluate.py --model rnn_only
```

### Run XGBoost

```bash
!python xgboost_baseline.py
```

### Run AdaBoost

```bash
!python adaboost_baseline.py
```

### Run Grad-CAM

```bash
!python gradcam_utils.py
```

---

# 24. What Each Python File Does

The monkey can think of the project files as different workers.

| File                   | Job                                                                |
| ---------------------- | ------------------------------------------------------------------ |
| `build_dataset.py`     | Prepare audio, augment it, extract features and create the dataset |
| `train.py`             | Train the neural networks                                          |
| `evaluate.py`          | Test a trained neural network and calculate metrics                |
| `xgboost_baseline.py`  | Train/test XGBoost                                                 |
| `adaboost_baseline.py` | Train/test AdaBoost                                                |
| `gradcam_utils.py`     | Generate Grad-CAM explanations                                     |
| `config.py`            | Store important project settings                                   |

So instead of thinking:

> "There are 7 complicated Python files."

Think:

> **Each file has one main job.**

---

# 25. The Most Important Thing to Understand

We did **not** manually type hundreds of commands to process every audio file.

We wrote the Python programs once.

Then Colab ran those programs.

For example:

```bash
!python build_dataset.py
```

means:

> "Hey Python, run the program called `build_dataset.py`."

That program then processed thousands of audio files automatically.

Likewise:

```bash
!python train.py
```

means:

> "Run the training program."

And:

```bash
!python evaluate.py --model hybrid
```

means:

> "Run the evaluation program and tell it to evaluate the hybrid model."

---

# 26. The Simplest Possible Explanation

If the monkey remembers only one thing, it should remember this:

### Step 1 — Prepare

```bash
!python build_dataset.py
```

**Turn the huge collection of audio files into prepared training data.**

### Step 2 — Train

```bash
!python train.py
```

**Teach the neural networks using that prepared data.**

### Step 3 — Test

```bash
!python evaluate.py --model hybrid
!python evaluate.py --model cnn_only
!python evaluate.py --model rnn_only
```

**Give the models unseen test data and measure how well they perform.**

### Step 4 — Compare

```bash
!python xgboost_baseline.py
!python adaboost_baseline.py
```

**Test two traditional machine-learning approaches for comparison.**

### Step 5 — Explain

```bash
!python gradcam_utils.py
```

**Show which parts of the sound representation influenced CNN-based decisions.**

So the whole project can be reduced to:

> **Prepare → Train → Test → Compare → Explain**

And Google Colab was simply the computer we used to run all of this.
