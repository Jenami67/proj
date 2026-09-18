# Deepfake Voice Detection Project

## Work Completed So Far

### 1. Project Overview

The purpose of this project is to build a system that listens to a voice recording and decides whether it is:

* **Real** – spoken by a real human.
* **Fake** – generated or modified using AI.

The basic process is:

> **Audio → Preprocessing → Augmentation → Feature Extraction → Model Training → Evaluation**

The project uses audio-processing libraries such as **librosa**, deep-learning libraries such as **TensorFlow and Keras**, and traditional machine-learning libraries such as **scikit-learn** and **XGBoost**.

The thesis proposes using spectral audio features and a hybrid CNN-RNN deep-learning architecture for this task.

The project has currently progressed through:

1. Dataset preparation
2. Dataset splitting
3. Audio preprocessing
4. Data augmentation
5. Feature extraction
6. Training three neural-network models
7. Training two traditional machine-learning baselines
8. Model checkpointing and resume support
9. Model evaluation
10. Grad-CAM explainability implementation

The exact file names may depend on the final project folder structure. In this report, the implementation is described using the main functional files:

* `dataset_builder.py`
* `preprocessing.py`
* `augmentation.py`
* `feature_extraction.py`
* `models.py`
* `train.py`
* `evaluate.py`
* `explainability.py`

---

# 2. Dataset

The main dataset used by the implemented system is:

**ASVspoof 2019 Logical Access (LA)**.

The dataset was downloaded as the `LA.zip` file from Kaggle using the provided Kaggle storage URL in Google Colab.

### Implementation file

The dataset download and directory setup were implemented in:

```text
dataset_builder.py
```

### Libraries used

The main libraries used in this phase are:

* `os` for creating and checking directories
* `wget` in Google Colab for downloading the dataset
* `zipfile` or shell extraction commands for extracting the archive
* Google Drive for permanent storage

The following code creates the dataset directory and downloads the dataset if it is not already available:

```python
LA_ZIP_URL = "https://storage.googleapis.com/kaggle-data-sets/8479661/13367197/bundle/archive.zip?..."

import os

dataset_dir = f"{PROJECT_DIR}/dataset"
os.makedirs(dataset_dir, exist_ok=True)

zip_path = f"{dataset_dir}/LA.zip"

if not os.path.exists(zip_path):
    assert LA_ZIP_URL != "PASTE_THE_COPIED_LINK_HERE"
    !wget -O {zip_path} "{LA_ZIP_URL}"
else:
    print("LA.zip already present, skipping download.")
```

The dataset contains:

* Real human speech
* AI-generated or spoofed speech
* Different speakers
* Multiple speech-generation methods

The ASVspoof dataset is organized into official partitions, normally including:

* Training
* Development
* Evaluation

Each partition contains audio files and a corresponding protocol text file.

The audio files and labels are stored separately. Therefore, the system reads the protocol file and matches each protocol entry with the correct `.flac` audio file.

A protocol row may look similar to:

```text
speaker_id file_id - attack_id bonafide
```

or:

```text
speaker_id file_id - attack_id spoof
```

The final label is converted as follows:

* `bonafide` → **real = 0**
* `spoof` → **fake = 1**

The protocol file is the source of truth for the labels. The system does not decide the label only from the filename or folder name.

The dataset-building process is:

1. Read a row from the protocol file.
2. Extract the speaker ID.
3. Extract the audio-file ID.
4. Extract the label.
5. Convert the label into a numerical value.
6. Find the matching audio file.
7. Load and process the audio.
8. Save the processed features and label.

Approximately:

**121,461 audio files**

were processed, including approximately:

* **12,483 real recordings**
* **108,978 fake recordings**

The dataset is highly unbalanced because there are many more fake recordings than real recordings. Class weighting and several evaluation metrics are therefore used during training and testing.

---

# 3. Dataset Splitting

The dataset was divided into:

* **60% training**
* **20% validation**
* **20% testing**

### Implementation file

The splitting process was implemented in:

```text
dataset_builder.py
```

### Libraries used

The main libraries used are:

* `pandas` for storing file information in tables
* `numpy` for numerical operations
* `scikit-learn` for splitting the dataset
* `GroupShuffleSplit` or similar grouped splitting methods for speaker-independent splitting

The split is designed to prevent the same speaker from appearing in more than one group.

---

## 3.1 Speaker-independent splitting

The speaker ID from the ASVspoof protocol file is used as the grouping value.

For example, if one speaker has 100 recordings, all of those recordings are placed in only one split:

* Training, or
* Validation, or
* Testing

The system does not divide the same speaker across all three groups.

A simplified example is:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(files, labels, groups=speaker_ids)
)
```

The code also checks that there is no speaker overlap:

```python
assert set(train_speakers).isdisjoint(validation_speakers)
assert set(train_speakers).isdisjoint(test_speakers)
assert set(validation_speakers).isdisjoint(test_speakers)
```

This is called a **speaker-independent split**.

It is important because the model should learn the difference between real and fake speech, not simply recognize a speaker it has already heard during training.

---

# 4. Audio Preprocessing

Before augmentation and feature extraction, all audio files are cleaned and standardized.

### Implementation file

The preprocessing pipeline was implemented in:

```text
preprocessing.py
```

### Main library used

The main library used is:

* **librosa** for loading, resampling and trimming audio
* **numpy** for numerical operations

The implemented pipeline is:

> **Audio → 16 kHz mono → silence trimming → pre-emphasis → normalization → fixed 4-second length**

A simplified version is:

```python
import librosa
import numpy as np

audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)

audio, _ = librosa.effects.trim(
    audio,
    top_db=40
)

audio = np.append(
    audio[0],
    audio[1:] - 0.97 * audio[:-1]
)

max_value = np.max(np.abs(audio))

if max_value > 0:
    audio = audio / max_value

target_length = 16000 * 4

if len(audio) < target_length:
    audio = np.pad(
        audio,
        (0, target_length - len(audio))
    )
else:
    start = (len(audio) - target_length) // 2
    audio = audio[start:start + target_length]
```

---

## 4.1 Convert to 16 kHz mono

All recordings are converted to:

* **16,000 Hz sampling rate**
* **Mono audio**

This gives every recording the same basic format.

The conversion is performed by `librosa.load()`:

```python
audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)
```

---

## 4.2 Remove silence

Silence at the beginning and end of the recording is removed.

The project uses an amplitude-based trimming method. The purpose is to focus on speech instead of empty parts of the recording.

This is implemented using:

```python
audio, _ = librosa.effects.trim(
    audio,
    top_db=40
)
```

---

## 4.3 Pre-emphasis

A pre-emphasis filter with:

**α = 0.97**

is applied.

The purpose is to slightly emphasize higher-frequency parts of the speech signal.

The implementation is:

```python
alpha = 0.97

audio = np.append(
    audio[0],
    audio[1:] - alpha * audio[:-1]
)
```

---

## 4.4 Normalization

The audio amplitude is normalized so that the largest absolute value is approximately 1.

```python
max_value = np.max(np.abs(audio))

if max_value > 0:
    audio = audio / max_value
```

This makes the volume level more consistent between recordings.

---

## 4.5 Fixed audio duration

Every recording is converted to:

**4 seconds**

Short recordings are padded with zeros.

Long recordings are center-cropped.

```python
target_length = 16000 * 4

if len(audio) < target_length:
    audio = np.pad(
        audio,
        (0, target_length - len(audio))
    )
else:
    start = (len(audio) - target_length) // 2
    audio = audio[start:start + target_length]
```

The thesis describes 4 seconds for shorter recordings and a central 6 seconds for longer recordings. The implemented project uses a fixed 4-second duration for all recordings. This difference should be reported honestly.

---

# 5. Data Augmentation

Augmentation is applied **before feature extraction**.

This order is important because augmentation changes the audio waveform. The changed waveform must then be used to calculate the MFCC, Mel-spectrogram and chroma features.

The implemented order is:

> **Audio → Preprocessing → Augmentation → Feature Extraction**

### Implementation file

The augmentation methods were implemented in:

```text
augmentation.py
```

### Libraries used

The main libraries used are:

* `numpy` for generating noise and changing waveform values
* `librosa` for speed and pitch changes
* `soundfile` or similar audio utilities for background-noise loading

The implemented augmentation techniques include:

* Gaussian noise
* Background noise
* Speed changes
* Pitch changes
* Compression-style noise

A simplified example is:

```python
import numpy as np
import librosa

def add_gaussian_noise(audio, noise_level=0.005):
    noise = np.random.normal(
        0,
        noise_level,
        size=audio.shape
    )
    return audio + noise

def change_speed(audio, rate=1.05):
    return librosa.effects.time_stretch(
        audio,
        rate=rate
    )

def change_pitch(audio, sr, steps=2):
    return librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=steps
    )
```

The augmentation is applied only to training data. Validation and test data are kept unchanged so that evaluation measures the model on original recordings.

The current implementation applies augmentation after loading and preprocessing the waveform but before calculating the features.

Therefore:

> **Thesis:** augmentation during training
> **Implementation:** augmentation before training, followed by feature caching

This is a deliberate simplification.

---

# 6. Feature Extraction

Feature extraction converts the audio waveform into numerical information that the models can use.

### Implementation file

The feature extraction process was implemented in:

```text
feature_extraction.py
```

### Main library used

The main library is:

* **librosa**

Other libraries used include:

* `numpy` for array operations
* `soundfile` for audio-related file handling
* `h5py` or NumPy files for saving features to disk

The feature extraction order is:

> **Augmented waveform → MFCC → Mel-spectrogram → Chroma → Save features**

A simplified version is:

```python
import librosa
import numpy as np

def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    mfcc_39 = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_chroma=12
    )

    return mfcc_39, mel_db, chroma
```

The extracted features are saved to disk so that they do not need to be recalculated every time the models are trained.

---

# 7. MFCC

MFCC stands for:

**Mel-Frequency Cepstral Coefficients**

MFCC provides a compact numerical description of speech characteristics.

### Implementation file

MFCC extraction was implemented in:

```text
feature_extraction.py
```

### Library used

The main library is:

* `librosa`

The project extracts:

* 13 normal MFCC coefficients
* 13 delta coefficients
* 13 delta-delta coefficients

This gives:

**13 + 13 + 13 = 39 coefficients**

The implementation is:

```python
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

delta = librosa.feature.delta(mfcc)

delta_delta = librosa.feature.delta(
    mfcc,
    order=2
)

mfcc_39 = np.concatenate(
    [mfcc, delta, delta_delta],
    axis=0
)
```

The final MFCC representation has 39 values for each time frame.

---

# 8. Mel-Spectrogram

A Mel-spectrogram shows how the frequency content of the audio changes over time.

### Implementation file

Mel-spectrogram extraction was implemented in:

```text
feature_extraction.py
```

### Library used

The main library is:

* `librosa`

The project uses:

* **128 Mel bands**
* A logarithmic decibel scale

```python
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)
```

The Mel-spectrogram is useful for the CNN because it can be treated like an image:

* Horizontal direction → time
* Vertical direction → frequency
* Values → sound strength

---

# 9. Chroma

Chroma describes the distribution of pitch classes in the audio.

### Implementation file

Chroma extraction was implemented in:

```text
feature_extraction.py
```

### Library used

The main library is:

* `librosa`

The project extracts 12 chroma values:

```python
chroma = librosa.feature.chroma_stft(
    y=audio,
    sr=sr,
    n_chroma=12
)
```

Although chroma is commonly used in music analysis, it can also provide information about pitch characteristics in speech.

---

# 10. Feature Caching and Memory-Mapped Storage

The extracted features are saved to disk instead of being kept entirely in memory.

### Implementation files

This part was implemented in:

```text
feature_extraction.py
train.py
```

### Libraries used

The main libraries are:

* `numpy`
* `tensorflow`
* `h5py` or NumPy memory-mapped files
* `tf.data`

The dataset is large, and Mel-spectrogram features can require several gigabytes of storage. Loading everything into RAM caused memory problems in Google Colab.

The project therefore uses memory-mapped NumPy files and gradual data loading.

A simplified example is:

```python
features = np.memmap(
    "mel_features.dat",
    dtype="float32",
    mode="r",
    shape=(num_samples, 128, time_steps)
)
```

During training, samples are loaded gradually using:

```python
dataset = tf.data.Dataset.from_generator(
    sample_generator,
    output_signature=output_signature
)
```

This means the system loads only the required samples instead of loading the entire dataset into RAM.

---

# 11. Handling the Class Imbalance

The dataset contains many more fake recordings than real recordings.

This is called **class imbalance**.

### Implementation files

Class weighting was implemented in:

```text
train.py
```

and for traditional models in:

```text
traditional_models.py
```

### Libraries used

The main libraries are:

* `scikit-learn`
* `numpy`
* `tensorflow`
* `keras`

For neural networks, balanced class weights can be calculated using:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)
```

The class weights are then passed to Keras during training:

```python
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    class_weight=class_weights
)
```

The traditional models also use class-weighting options suitable for those algorithms.

Class weighting helps prevent the models from simply predicting the most common class.

---

# 12. Models Tested

Five models have been implemented and evaluated:

1. CNN-only
2. RNN-only
3. Hybrid CNN-RNN
4. XGBoost
5. AdaBoost

### Implementation files

The deep-learning models were implemented in:

```text
models.py
```

The traditional models were implemented in:

```text
traditional_models.py
```

### Libraries used

The main libraries are:

* **TensorFlow**
* **Keras**
* **scikit-learn**
* **XGBoost**

The deep-learning models use structured audio features such as:

* Mel-spectrograms
* MFCC sequences
* Chroma features

The traditional models use summarized numerical features.

---

# 13. CNN-only Model

CNN means:

**Convolutional Neural Network**

The CNN-only model receives the Mel-spectrogram.

### Implementation file

The model was implemented in:

```text
models.py
```

### Libraries used

The main libraries are:

* `tensorflow`
* `keras`

A simplified structure is:

```python
from tensorflow.keras import layers, models

def build_cnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128,
        kernel_size=3,
        activation="relu"
    )(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        2,
        activation="softmax"
    )(x)

    return models.Model(inputs, outputs)
```

The CNN contains convolutional layers with:

* 32 filters
* 64 filters
* 128 filters

It also uses:

* Batch Normalization
* Max Pooling
* Global Average Pooling
* Dense classification layers

In simple terms:

> **CNN = look at the sound picture and find suspicious patterns.**

---

# 14. RNN-only Model

RNN means:

**Recurrent Neural Network**

The RNN-only model receives the MFCC sequence.

### Implementation file

The model was implemented in:

```text
models.py
```

### Libraries used

The main libraries are:

* `tensorflow`
* `keras`

The project uses bidirectional LSTM layers.

A simplified structure is:

```python
from tensorflow.keras import layers, models

def build_rnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(64)
    )(x)

    outputs = layers.Dense(
        2,
        activation="softmax"
    )(x)

    return models.Model(inputs, outputs)
```

The RNN learns how speech features change over time.

In simple terms:

> **RNN/LSTM = listen to how the voice changes over time.**

---

# 15. Hybrid CNN-RNN Model

The hybrid CNN-RNN model is the main proposed model.

### Implementation file

The model was implemented in:

```text
models.py
```

### Libraries used

The main libraries are:

* `tensorflow`
* `keras`

The model has two branches:

### CNN branch

Receives the Mel-spectrogram and learns time-frequency patterns.

### RNN branch

Receives the MFCC sequence and learns temporal patterns.

The two branches are combined using a concatenation layer.

A simplified structure is:

```python
from tensorflow.keras import layers, models

def build_hybrid_model(
    mel_shape,
    mfcc_shape
):
    mel_input = layers.Input(
        shape=mel_shape,
        name="mel_input"
    )

    mfcc_input = layers.Input(
        shape=mfcc_shape,
        name="mfcc_input"
    )

    cnn = layers.Conv2D(
        32,
        3,
        activation="relu"
    )(mel_input)

    cnn = layers.MaxPooling2D()(cnn)
    cnn = layers.Conv2D(
        64,
        3,
        activation="relu"
    )(cnn)

    cnn = layers.GlobalAveragePooling2D()(cnn)

    rnn = layers.Bidirectional(
        layers.LSTM(128)
    )(mfcc_input)

    combined = layers.Concatenate()(
        [cnn, rnn]
    )

    x = layers.Dense(
        128,
        activation="relu"
    )(combined)

    outputs = layers.Dense(
        2,
        activation="softmax"
    )(x)

    return models.Model(
        [mel_input, mfcc_input],
        outputs
    )
```

The process is:

> **CNN information + RNN information → Fusion → Dense layers → Real/Fake**

In simple terms:

> **CNN:** “What does the sound look like?”
> **RNN:** “How does the sound change over time?”
> **Hybrid:** “Let us use both types of information.”

---

# 16. XGBoost Baseline

XGBoost is a traditional machine-learning algorithm.

### Implementation file

The XGBoost model was implemented in:

```text
traditional_models.py
```

### Libraries used

The main libraries are:

* `xgboost`
* `scikit-learn`
* `numpy`

XGBoost expects a fixed list of numerical values. Therefore, the project summarizes the audio features using values such as:

* Mean
* Standard deviation

A simplified example is:

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)
```

The MFCC, Mel-spectrogram and chroma summaries are combined into one numerical vector.

In simple terms:

> **XGBoost = use many prepared measurements from the audio and classify them with a traditional algorithm.**

---

# 17. AdaBoost Baseline

AdaBoost is another traditional machine-learning algorithm.

### Implementation file

The AdaBoost model was implemented in:

```text
traditional_models.py
```

### Libraries used

The main libraries are:

* `scikit-learn`
* `numpy`

A simplified example is:

```python
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(
    n_estimators=200,
    random_state=42
)

ada_model.fit(
    X_train,
    y_train
)
```

The current AdaBoost implementation uses summarized:

* MFCC
* Mel
* Chroma

features.

The thesis proposes an AdaBoost baseline based on biological or pause-related speech characteristics. However, the current implementation does not use those exact pause features.

Therefore, the correct description is:

> **AdaBoost was implemented as a comparative baseline using aggregated audio features.**

---

# 18. Training

The training process was implemented in:

```text
train.py
```

### Libraries used

The main libraries are:

* **TensorFlow**
* **Keras**
* `numpy`
* `scikit-learn`

The neural networks were trained using:

* Optimizer: **Adam**
* Learning rate: **0.001**
* Loss function: **Categorical Cross-Entropy**
* Batch size: **32**
* Maximum epochs: **50**
* Early stopping patience: **8 epochs**

A simplified training setup is:

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50,
    callbacks=[early_stopping],
    class_weight=class_weights
)
```

Early stopping prevents unnecessary training after the validation performance stops improving.

---

# 19. Checkpoints and Resume Support

Checkpointing was implemented in:

```text
train.py
```

### Libraries used

The main library is:

* **Keras**

The project saves the best model during training.

Important model files include:

* `hybrid_cnn_rnn_best.keras`
* `cnn_only_best.keras`
* `rnn_only_best.keras`

The final models are also saved.

A simplified checkpoint callback is:

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "hybrid_cnn_rnn_best.keras",
    monitor="val_loss",
    save_best_only=True
)
```

Training progress is saved in JSON files.

This was especially useful because the experiments were run on the free tier of Google Colab, where sessions can disconnect or stop.

The project can resume from a saved checkpoint instead of starting from the beginning.

---

# 20. Memory Optimization

Memory optimization was implemented in:

```text
feature_extraction.py
train.py
```

### Libraries used

The main libraries are:

* `numpy`
* `tensorflow`
* `tf.data`
* `h5py` or NumPy memory-mapped files

The dataset is very large, and the free Google Colab environment has limited RAM.

The project uses:

* Memory-mapped NumPy files
* Feature caching
* `tf.data.Dataset.from_generator()`
* Gradual sample loading

A simplified example is:

```python
dataset = tf.data.Dataset.from_generator(
    sample_generator,
    output_signature=output_signature
)
```

This avoids loading the entire dataset into memory.

In simple terms:

> Instead of putting the whole dataset into RAM, the system keeps it on disk and loads only the samples needed for training.

---

# 21. Evaluation

The evaluation process was implemented in:

```text
evaluate.py
```

### Libraries used

The main libraries are:

* `scikit-learn`
* `numpy`
* `tensorflow`
* `keras`

The models are evaluated using the separate test set.

The evaluation code calculates:

* Accuracy
* Precision
* Recall
* Specificity
* F1-score
* ROC-AUC
* Equal Error Rate
* Confusion matrix
* Inference time

A simplified example is:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_probability
)

matrix = confusion_matrix(
    y_true,
    y_pred
)
```

---

## 21.1 Accuracy

Accuracy measures how many predictions were correct overall.

> “Out of all recordings, how often was the model correct?”

---

## 21.2 Precision

Precision answers:

> “When the model says a recording is fake, how often is it actually fake?”

---

## 21.3 Recall

Recall answers:

> “Out of all fake recordings, how many did the model find?”

---

## 21.4 Specificity

Specificity answers:

> “Out of all real recordings, how many did the model correctly identify as real?”

---

## 21.5 F1-score

F1-score combines precision and recall.

It is useful when the dataset is unbalanced.

---

## 21.6 ROC-AUC

ROC-AUC measures how well the model separates real and fake recordings across different decision thresholds.

A value closer to 1 is generally better.

---

## 21.7 Equal Error Rate

Equal Error Rate, or EER, is the point where false-acceptance and false-rejection rates are equal.

A lower EER is generally better.

---

# 22. Current Evaluation Results

The current test results are:

| Model          | Accuracy | Precision | Recall | Specificity |     F1 | ROC-AUC |    EER |
| -------------- | -------: | --------: | -----: | ----------: | -----: | ------: | -----: |
| CNN-only       |   97.09% |    98.71% | 98.07% |      87.69% | 98.39% |  99.06% |  4.94% |
| Hybrid CNN-RNN |   96.50% |    99.06% | 97.06% |      91.19% | 98.05% |  99.09% |  4.90% |
| RNN-only       |   86.03% |    97.29% | 87.00% |      76.82% | 91.86% |  91.06% | 17.45% |
| XGBoost        |   86.52% |    97.14% | 87.69% |      75.26% | 92.17% |  90.84% | 17.58% |
| AdaBoost       |   66.60% |    97.18% | 65.00% |      81.97% | 77.90% |  81.36% | 26.29% |

These values come from the current evaluation JSON files.

At this stage, the values are recorded as experimental results. A detailed explanation of why the models performed differently will be provided separately.

---

# 23. Confusion Matrix Results

The confusion matrix results are:

| Model          | True Negative | False Positive | False Negative | True Positive |
| -------------- | ------------: | -------------: | -------------: | ------------: |
| CNN-only       |         2,130 |            299 |            448 |        22,790 |
| Hybrid CNN-RNN |         2,215 |            214 |            684 |        22,554 |
| RNN-only       |         1,866 |            563 |          3,022 |        20,216 |
| XGBoost        |         1,828 |            601 |          2,860 |        20,378 |
| AdaBoost       |         1,991 |            438 |          8,134 |        15,104 |

The four terms mean:

### True Positive

The recording was fake and the model correctly predicted fake.

### True Negative

The recording was real and the model correctly predicted real.

### False Positive

The recording was real but the model incorrectly predicted fake.

### False Negative

The recording was fake but the model incorrectly predicted real.

---

# 24. Inference Speed

Inference means the time required by the trained model to make a prediction.

### Implementation file

Inference-time measurement was implemented in:

```text
evaluate.py
```

### Libraries used

The main libraries are:

* `tensorflow`
* `keras`
* `time`
* `numpy`

The current neural-network evaluation reported:

| Model          |          Inference |
| -------------- | -----------------: |
| CNN-only       | 0.392 ms/sec audio |
| Hybrid CNN-RNN | 0.685 ms/sec audio |
| RNN-only       | 0.155 ms/sec audio |

These values were calculated by measuring prediction time on the test set and dividing it by the total duration of the test audio.

---

# 25. Grad-CAM Explainability

Grad-CAM was implemented in:

```text
explainability.py
```

### Libraries used

The main libraries are:

* **TensorFlow**
* **Keras**
* `numpy`
* `matplotlib`

Grad-CAM creates a heatmap showing which parts of the Mel-spectrogram influenced the model's decision.

For example, instead of only showing:

> “Fake: 98%”

the system can also show which time-frequency areas contributed most to the prediction.

A simplified Grad-CAM process is:

```python
with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(
        model_input
    )
    class_score = predictions[:, class_index]

gradients = tape.gradient(
    class_score,
    conv_output
)

weights = tf.reduce_mean(
    gradients,
    axis=(1, 2)
)

heatmap = tf.reduce_sum(
    conv_output * weights[:, None, None, :],
    axis=-1
)
```

Grad-CAM supports:

* Hybrid CNN-RNN
* CNN-only

It is not directly applied to the RNN-only model because the RNN-only model does not contain a convolutional layer.

The thesis identifies Grad-CAM and LRP as possible explainability methods. The current implementation uses **Grad-CAM**, not LRP.

---

# 26. Files and Results Saved

The project saves models, training progress, evaluation results and experiment information.

### Implementation files

Saving and file management were implemented in:

```text
train.py
evaluate.py
```

### Libraries used

The main libraries are:

* `os`
* `json`
* `tensorflow`
* `keras`
* Google Drive tools in Google Colab

Important model files include:

* `hybrid_cnn_rnn_best.keras`
* `hybrid_cnn_rnn_final.keras`
* `cnn_only_best.keras`
* `cnn_only_final.keras`
* `rnn_only_best.keras`
* `rnn_only_final.keras`

Training progress is saved in:

* `hybrid_training_progress.json`
* `cnn_only_training_progress.json`
* `rnn_only_training_progress.json`

Evaluation results are saved in:

* `hybrid_eval.json`
* `cnn_only_eval.json`
* `rnn_only_eval.json`
* `xgboost_eval.json`
* `adaboost_eval.json`

The processed datasets, cached features, checkpoints and result files are stored in Google Drive so they remain available after a Colab session ends.

---

# 27. What Has Been Successfully Completed

## Dataset

✓ ASVspoof 2019 LA processed
✓ Real/fake labels extracted from official protocols
✓ Approximately 121,461 files processed
✓ Speaker IDs used during splitting
✓ Speaker-overlap check implemented

## Preprocessing

✓ 16 kHz sampling rate
✓ Mono audio
✓ Silence trimming
✓ Pre-emphasis
✓ Amplitude normalization
✓ Fixed 4-second duration

## Augmentation

✓ Gaussian noise
✓ Background noise support
✓ Speed perturbation
✓ Pitch shifting
✓ Compression-style augmentation
✓ Applied to training data only
✓ Applied before feature extraction

## Features

✓ 39-dimensional MFCC representation
✓ 128-band Mel-spectrogram
✓ 12-dimensional chroma representation
✓ Feature caching to disk

## Deep Learning

✓ CNN-only model
✓ RNN-only model
✓ Hybrid CNN-RNN model
✓ CNN/RNN feature fusion
✓ Class weighting
✓ Adam optimizer
✓ Early stopping
✓ Best-model checkpointing
✓ Training resume support

## Traditional Machine Learning

✓ XGBoost baseline
✓ AdaBoost baseline
✓ Class-imbalance handling

## Evaluation

✓ Accuracy
✓ Precision
✓ Recall
✓ Specificity
✓ F1-score
✓ ROC-AUC
✓ EER
✓ Confusion matrix
✓ Inference-time measurement

## Explainability

✓ Grad-CAM implemented
✓ CNN-only support
✓ Hybrid-model support
✓ Heatmap generation

---

# 28. What Has Not Yet Been Fully Completed

Several items described in the thesis have not yet been implemented or fully demonstrated in the current experiments.

These should not be claimed as completed until they are implemented and tested.

Examples include:

* WaveFake cross-dataset testing
* VoiceWukong final evaluation
* AASIST2 comparison
* 1,000 bootstrap confidence intervals
* McNemar's statistical test
* Cohen's d effect-size analysis
* Full transfer-learning or adaptation experiments
* Full LRP explainability
* Production deployment or API
* Complete real-time application testing

The thesis proposes these components as part of the research methodology, but the current code and results do not show that all of them have been completed.

---

# 29. Current Project Status

The project has reached an important milestone:

> **The complete basic experimental pipeline is working.**

The system can:

1. Download and organize the ASVspoof dataset.
2. Read labels from the official protocol files.
3. Split the data by speaker.
4. Preprocess the audio.
5. Apply augmentation to training audio.
6. Extract MFCC, Mel-spectrogram and chroma features.
7. Save the features to disk.
8. Train multiple deep-learning and traditional models.
9. Save checkpoints and final models.
10. Evaluate the models using several metrics.
11. Generate Grad-CAM visualizations for CNN-based models.

The main implemented processing order is:

> **Dataset → Splitting → Preprocessing → Augmentation → Feature Extraction → Training → Evaluation → Explainability**

The five main approaches have been tested under the same evaluation framework.

This provides the foundation for the next stage of the project: detailed analysis of the results.

The next analysis should answer questions such as:

* Why did CNN perform so well?
* Why did the hybrid model behave differently?
* What does the difference between CNN and hybrid models mean?
* Why did the RNN perform worse?
* Why did XGBoost behave similarly to RNN?
* Why did AdaBoost perform considerably worse?
* What do the false positives and false negatives show?
* Does the model satisfy the thesis requirements?
* Are the results genuinely strong, or could there be a methodological problem?
* Which research questions can now be answered?
* Which experiments are still necessary before the project can be considered complete?
