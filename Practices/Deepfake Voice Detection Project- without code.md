# Deepfake Voice Detection Project

## Work Completed So Far

### 1. Project Overview

The purpose of this project is to build a system that can listen to a voice recording and decide whether it is:

* **Real** – spoken by a real human.
* **Fake** – generated or modified using AI.

The basic idea is very simple:

> **Give the system audio → turn the audio into useful information → let AI learn the difference between real and fake voices → test how well it can tell them apart.**

The thesis proposes using spectral audio features and a hybrid CNN-RNN deep learning architecture for this task.

The project has currently progressed through:

1. Dataset preparation
2. Audio preprocessing
3. Feature extraction
4. Data augmentation
5. Speaker-independent dataset splitting
6. Training three neural-network models
7. Training two traditional machine-learning baselines
8. Model checkpointing and resume support
9. Model evaluation
10. Grad-CAM explainability implementation

---

# 2. Dataset

The main dataset currently used by the implemented system is:

**ASVspoof 2019 Logical Access (LA)**.

The ASVspoof 2019 Logical Access (LA) dataset was downloaded as the `LA.zip` file from Kaggle using the provided Kaggle storage URL in Google Colab. The following code creates the dataset directory, checks whether `LA.zip` is already present, and downloads it with `wget` if necessary:

```python
LA_ZIP_URL = "https://storage.googleapis.com/kaggle-data-sets/8479661/13367197/bundle/archive.zip?..."

import os

dataset_dir = f'{PROJECT_DIR}/dataset'
os.makedirs(dataset_dir, exist_ok=True)
zip_path = f'{dataset_dir}/LA.zip'

if not os.path.exists(zip_path):
    assert LA_ZIP_URL != "PASTE_THE_COPIED_LINK_HERE", "You forgot to paste the real link!"
    !wget -O {zip_path} "{LA_ZIP_URL}"
else:
    print("LA.zip already present, skipping download.")
```

After downloading, the `LA.zip` archive was stored in the project dataset directory for subsequent extraction and processing. Because Google Colab storage is temporary and can be wiped when a session ends, Google Drive was used to preserve the dataset, code files, extracted features, trained models, results, and other project files across sessions.

The dataset contains:

* Real human speech
* AI-generated/spoofed speech
* Different speakers
* Multiple speech-generation methods

The thesis identifies ASVspoof 2019 LA as one of the main benchmark datasets for the project.

The project used the ASVspoof 2019 Logical Access (LA) dataset. ASVspoof is not just one folder of audio files. It is organized into separate official partitions, normally including:

* A training set
* A development set
* An evaluation set

Each partition contains audio recordings and a corresponding protocol text file. The audio files are stored separately from the labels. This means that the label is not necessarily written directly into the audio filename. Instead, the system must read the correct protocol file and match each protocol entry to the corresponding audio file.

The protocol file is a plain-text `.txt` file in which each row describes one audio recording. The row contains information such as:

* The speaker or source identifier
* The audio-file identifier
* Additional metadata fields
* The attack or spoofing information
* The final class label, such as `bonafide` or `spoof`

The exact position of some fields depends on the ASVspoof protocol format, so the dataset-building code reads the protocol columns and extracts the relevant file identifier and label. The file identifier is then used to locate the matching `.flac` audio file in the appropriate ASVspoof partition.

For example, a protocol row may contain an entry similar to:

```text
speaker_id file_id - attack_id bonafide
```

or:

```text
speaker_id file_id - attack_id spoof
```

The important part for this project is the final class label:

* `bonafide` means the recording is genuine human speech and is converted to **real = 0**
* `spoof` means the recording is generated, converted or manipulated speech and is converted to **fake = 1**

The protocol file is therefore the source of truth for the labels. The system does not decide that a file is real or fake based only on its filename, folder name or audio content. Instead, it performs the following process:

1. Read a row from the ASVspoof protocol text file.
2. Extract the audio-file identifier.
3. Extract the label from the protocol row.
4. Convert `bonafide` or `spoof` into the numerical class used by the neural networks.
5. Search for the matching audio file in the ASVspoof directory.
6. Load and preprocess that audio file.
7. Save the audio features together with the protocol-derived label.

This is important because ASVspoof uses a controlled experimental structure. The spoofed recordings are associated with different attack types, representing different methods of generating or modifying speech. The protocol file can therefore contain more information than just the final real/fake label. It can also identify the speaker, the recording, the attack condition and the dataset partition. In the current classification task, the attack information is not used as the prediction target. The system simplifies the problem into a binary decision:

> Is this recording `bonafide` or `spoof`?

The project combined the official ASVspoof training, development and evaluation partitions into one pool and then created its own split. This means that the original ASVspoof partition boundaries were not used as the final training, validation and testing boundaries in the current experiment. Instead, the recordings from the available official partitions were collected, matched with their protocol labels, grouped by speaker, and then divided into:

* 60% training
* 20% validation
* 20% testing

The speaker identifier from the protocol file was also retained for this process. This allowed the project to perform a speaker-independent split, meaning that recordings from the same speaker were kept in only one of the three new groups. This prevents the model from seeing the same speaker during training and testing.

The dataset processing successfully handled approximately:

**121,461 audio files**

consisting of approximately:

* **12,483 real**
* **108,978 fake**

The dataset is therefore highly unbalanced: there are many more fake recordings than real recordings. This imbalance is one reason why the project uses class weighting during model training and reports metrics such as precision, recall, specificity, F1-score, ROC-AUC and EER instead of relying only on accuracy.

---

# 3. Dataset Splitting

The dataset was divided into three groups:

* **60% Training**
* **20% Validation**
* **20% Testing**

This follows the methodology specified in the thesis.

Think of it like teaching a student:

### Training set

The student uses these examples to **learn**.

### Validation set

The student uses these examples to check:

> "Am I learning correctly?"

It is also used during training for things such as early stopping.

### Test set

This is the **final exam**.

The model does not use these samples to learn.

---

## 3.1 Speaker-independent splitting

One particularly important part of the project is that the same speaker is not allowed to appear in multiple groups.

For example, imagine:

> Ali has 100 recordings.

We don't want:

* 60 Ali recordings → training
* 20 Ali recordings → validation
* 20 Ali recordings → testing

because the model could partially recognize **Ali's voice** rather than learning what makes speech real or fake.

Instead, the splitting code uses the speaker ID as a group and verifies that there is no speaker overlap.

The code explicitly checks:

> **No speaker appears in more than one split.**

This is called a **speaker-independent split**.

The thesis specifically requires this approach to reduce speaker-specific overfitting and test performance on unseen voices.

---

# 4. Audio Preprocessing

Before giving audio to the AI, all recordings are cleaned and standardized.

Think of this as:

> "Make all the audio files look similar before we ask the AI to compare them."

The implemented preprocessing pipeline is:

**Audio → 16 kHz mono → silence trimming → pre-emphasis → normalization → 4-second length**

---

## 4.1 Convert to 16 kHz mono

All audio is converted to:

* **16,000 Hz sampling rate**
* **Mono**

This means every recording uses the same basic audio format.

---

## 4.2 Remove silence

Unnecessary silence at the beginning and end of recordings is removed.

The project uses an amplitude-based threshold corresponding to:

**0.01 × maximum amplitude**

The purpose is to focus the system more on actual speech rather than empty silence.

This follows the preprocessing specification in the thesis.

---

## 4.3 Pre-emphasis

A pre-emphasis filter with:

**α = 0.97**

is applied.

In simple words:

> It slightly emphasizes higher-frequency parts of the speech signal.

This is a traditional speech-processing step intended to make some useful speech characteristics easier to analyze.

---

## 4.4 Normalization

The audio amplitude is normalized to approximately:

**-1 to +1**

This prevents one recording from simply being much louder than another and makes the inputs more consistent.

---

## 4.5 Fixed audio duration

Every recording is ultimately converted to:

**4 seconds**

Shorter recordings are padded with zeros.

Longer recordings are center-cropped.

The thesis describes 4 seconds for shorter recordings and a central 6 seconds for longer recordings; the implemented project simplified this to a fixed 4-second duration for all recordings.

This difference should be reported honestly rather than claiming the implementation exactly matches the thesis.

---

# 5. Feature Extraction

This is one of the most important parts of the project.

The AI does not directly receive:

> "Here is a WAV file. Figure it out."

Instead, the audio is converted into numerical representations called **features**.

Think of features as:

> **Useful measurements extracted from the audio.**

The project extracts three main types:

1. MFCC
2. Mel-spectrogram
3. Chroma

The thesis specifically identifies these as important spectral features.

---

# 6. MFCC

MFCC stands for:

**Mel-Frequency Cepstral Coefficients**

Don't worry about memorizing the complicated name.

For this project, think of MFCC as:

> **A compact numerical description of the characteristics of human speech.**

The implementation extracts:

* 13 normal MFCC coefficients
* 13 delta coefficients
* 13 delta-delta coefficients

Giving:

**13 + 13 + 13 = 39 coefficients**

So each time frame of audio gets a 39-number description.

The 39-dimensional MFCC representation is consistent with the thesis requirement for MFCC features.

---

# 7. Mel-Spectrogram

A Mel-spectrogram represents the audio in a way that shows:

> **Which frequencies are present and how they change over time.**

You can think of it as turning sound into a picture.

For example:

* horizontal direction → time
* vertical direction → frequency
* values → strength of the sound

The project uses:

**128 Mel bands**

and converts the result to a logarithmic decibel scale.

This representation is particularly useful for the CNN because a CNN is very good at recognizing patterns in something that looks like an image.

---

# 8. Chroma

Chroma describes the distribution of musical pitch classes.

The implementation extracts:

**12 chroma values**

Although chroma is more commonly associated with music, it can also provide information about the pitch characteristics of speech.

The project therefore stores MFCC, Mel-spectrogram and chroma representations for the audio.

---

# 9. Data Augmentation

The training data is also modified in several ways to make the model less dependent on perfectly clean recordings.

The idea is:

> If the AI only sees perfect audio during training, it may struggle when someone gives it slightly noisy or altered audio.

The implemented augmentation techniques include:

* Gaussian noise
* Background noise
* Speed changes
* Pitch changes
* Compression-style noise

The probabilities are based on the percentages specified in the thesis.

However, there is an important implementation difference.

The thesis describes this as **online augmentation**, meaning new variations could be generated during training.

The current implementation applies augmentation to the audio waveform after loading the WAV file but before extracting the MFCC, Mel-spectrogram and chroma features. The augmented waveform is then converted into feature matrices, and those matrices are saved for training.

So:

> **Thesis:** augmentation during training
> **Implementation:** augmentation before training, then cached

This is a deliberate simplification in the current project.

---

# 10. Handling the Class Imbalance

The dataset does not contain the same number of real and fake audio recordings.

There are:

* Many more fake recordings
* Fewer real recordings

This is called **class imbalance**.

For example, imagine a dataset with:

* 90 fake recordings
* 10 real recordings

If the AI always predicts:

> "Fake"

then it will be correct 90% of the time.

That sounds like a high accuracy, but the AI has not really learned how to recognize real voices. It has simply learned to choose the most common class.

This would be a problem because correctly identifying real voices is also important.

To reduce this problem, the training process uses **class weighting**.

Class weighting tells the model:

> "Do not treat every mistake as equally unimportant. Pay more attention to mistakes involving the smaller class."

Because real recordings are less common, mistakes involving real recordings receive more importance during training.

The neural-network models use scikit-learn's balanced class-weight calculation to calculate these weights automatically.

The traditional machine-learning models also use class-weighting methods that are suitable for those models.

The purpose of class weighting is to stop the models from simply choosing the most common label.

It encourages the models to learn both classes:

* What real human speech sounds like
* What fake or generated speech sounds like

This is important because a model can have high overall accuracy while still performing badly on real voices.

Therefore, accuracy is not considered by itself. The project also checks other measurements, such as:

* Precision
* Recall
* Specificity
* F1-score
* ROC-AUC
* Equal Error Rate (EER)

These additional measurements help show whether the model is genuinely learning to separate real and fake voices, rather than only predicting the class that appears most often.

---

# 11. Models Tested

Five different models have currently been implemented and evaluated.

They can be divided into two groups:

1. **Deep-learning models**
2. **Traditional machine-learning models**

## 11.1 What is a baseline?

A **baseline** is a simpler or alternative model used as a reference point.

It helps answer:

> "Is the proposed model actually better than other reasonable approaches?"

For example, if the hybrid CNN-RNN model achieves 95% accuracy, that number is difficult to judge by itself. However, if:

* CNN-only achieves 90%
* RNN-only achieves 85%
* XGBoost achieves 82%
* AdaBoost achieves 80%
* Hybrid CNN-RNN achieves 95%

then we can compare the hybrid model against these reference models.

In this project, CNN-only, RNN-only, XGBoost and AdaBoost are comparison models, or baselines, for evaluating the proposed hybrid CNN-RNN model.

A baseline does not necessarily mean that the model is bad or unimportant. It simply provides a point of comparison.

## 11.2 Deep-learning models

The deep-learning models are:

1. CNN-only
2. RNN-only
3. Hybrid CNN-RNN

Deep learning uses neural networks with multiple layers. These networks can learn useful patterns directly from the input features during training.

For example, the model can learn which combinations of frequency patterns, speech characteristics and changes over time are associated with real or fake speech.

The deep-learning models in this project receive structured audio features such as:

* Mel-spectrograms
* MFCC sequences
* Chroma features

The models then learn which patterns are useful for distinguishing real speech from fake speech.

### CNN-only

The CNN-only model learns spatial or time-frequency patterns from the Mel-spectrogram.

### RNN-only

The RNN-only model learns how speech features change over time using MFCC sequences.

### Hybrid CNN-RNN

The hybrid model combines both types of information:

* The CNN examines the sound's time-frequency patterns.
* The RNN examines how the speech changes over time.

The two outputs are then combined before the final real/fake prediction.

## 11.3 Traditional machine-learning models

The traditional machine-learning models are:

4. XGBoost
5. AdaBoost

Traditional machine-learning models usually do not learn directly from large, complex feature structures such as complete spectrogram images or long sequences.

Instead, the audio features are first summarized into a fixed list of numerical values.

For example, the project calculates summary statistics such as:

* Mean
* Standard deviation

for the MFCC, Mel-spectrogram and chroma features.

These summary values are then given to XGBoost or AdaBoost.

This means that the traditional models rely more heavily on the features and summaries prepared before training, while the deep-learning models can learn more complex patterns from the feature representations.

## 11.4 Main difference between the two groups

The main difference is how much pattern learning is performed by the model itself.

### Traditional machine learning

The general process is:

**Audio → manually selected features → summary values → model → prediction**

The researcher decides which features and summary measurements should be provided to the model.

The model then learns from this fixed numerical table.

### Deep learning

The general process is:

**Audio → structured feature representation → neural network learns complex patterns → prediction**

The researcher still chooses the input representation, such as MFCC or Mel-spectrogram, but the neural network learns more complicated combinations and patterns from those inputs.

In simple terms:

> **Traditional machine learning:** give the model a prepared list of measurements.
> **Deep learning:** give the model a richer representation and let it learn more complex patterns from it.

## 11.5 Why both types of models are used

Using both groups makes the experiment more meaningful.

If only the hybrid CNN-RNN model were tested, it would be difficult to know whether the results came from:

* The CNN branch
* The RNN branch
* The combination of both
* The feature extraction process
* The general difficulty of the dataset

By comparing several models, the project can investigate:

* Whether CNN-based spectral analysis is useful
* Whether RNN-based temporal analysis is useful
* Whether combining CNN and RNN improves performance
* Whether deep-learning models outperform traditional approaches
* Whether the additional complexity of the hybrid model is justified

The thesis specifically calls for comparing the hybrid model with CNN-only, RNN-only, XGBoost and AdaBoost approaches.

The reason for having several models is simple:

> We don't want to build one model and say "it works."

We want to ask:

> **"Does combining CNN and RNN actually help?"**

---

# 12. CNN-only Model

CNN means:

**Convolutional Neural Network**

For this project, the CNN looks at the **Mel-spectrogram**.

Think of the Mel-spectrogram as a picture of the sound.

The CNN looks for patterns in that picture.

The implemented CNN contains several convolutional layers:

* 32 filters
* 64 filters
* 128 filters

with:

* Batch Normalization
* Max Pooling
* Global Average Pooling

The extracted information is then passed through dense layers to produce the final classification.

In simple terms:

> **CNN = look at the sound picture and find suspicious patterns.**

---

# 13. RNN-only Model

RNN means:

**Recurrent Neural Network**

The RNN is designed for sequences.

Speech is a sequence:

> sound → sound → sound → sound → ...

The RNN therefore receives the MFCC sequence and tries to understand how the characteristics of the voice change over time.

The project uses:

**Bidirectional LSTM**

layers.

LSTM is a type of RNN designed to remember useful information across a sequence.

The project uses two bidirectional LSTM layers.

In simple terms:

> **RNN/LSTM = listen to how the voice changes over time.**

---

# 14. Hybrid CNN-RNN Model

This is the main proposed model.

Instead of asking one model to do everything, it gives different jobs to different branches.

### CNN branch

Looks at:

**Mel-spectrogram**

and searches for spectral patterns.

### RNN branch

Looks at:

**MFCC sequence**

and searches for temporal patterns.

Then:

**CNN information + RNN information → Fusion → Dense layers → Real/Fake**

The two branches are combined using a concatenation/fusion layer.

The thesis describes the hybrid architecture as combining CNN-based spectral analysis with RNN-based temporal modeling.

So the monkey version is:

> **CNN:** "What does the sound look like?"
> **RNN:** "How does the sound change over time?"
> **Hybrid:** "Let's ask both."

---

# 15. XGBoost Baseline

XGBoost is a traditional machine-learning algorithm.

It is not a neural network.

It is included as a comparison point.

Because XGBoost expects a fixed list of numbers rather than a complete time sequence, the project summarizes each feature using:

* Mean
* Standard deviation

for each feature dimension.

MFCC + Mel + chroma are then combined into one large numerical vector.

XGBoost learns from this vector.

In simple terms:

> **XGBoost = take many useful measurements from the audio and use a traditional machine-learning algorithm to classify them.**

The thesis includes XGBoost as a temporal-feature baseline.

---

# 16. AdaBoost Baseline

AdaBoost is another traditional machine-learning algorithm.

The thesis proposes an AdaBoost baseline based on biological/pause-related speech characteristics.

However, the current implementation does **not** implement those exact pause features.

Instead, the current AdaBoost implementation uses the same summarized:

* MFCC
* Mel
* Chroma

features used for the XGBoost baseline.

Therefore, the current report should say:

> **AdaBoost was implemented as a comparative baseline using aggregated audio features.**

It should **not** claim that the current AdaBoost experiment reproduces the exact pause-feature method from Nair et al.

---

# 17. Training

The neural networks were trained using:

### Optimizer

**Adam**

with learning rate:

**0.001**

### Loss function

**Categorical Cross-Entropy**

### Batch size

**32**

### Maximum epochs

**50**

### Early stopping

Training stops early if validation loss does not improve for:

**8 epochs**

This prevents wasting training time after the model stops improving.

---

# 18. Checkpoints and Resume

Because training large audio datasets can take a long time, the project includes checkpoint support.

This was especially necessary because the experiments were run using the **free tier of Google Colab**, which provides limited computing resources and can disconnect or terminate sessions after a period of time.

During training, the best model is saved as:

* `hybrid_cnn_rnn_best.keras`
* `cnn_only_best.keras`
* `rnn_only_best.keras`

The final trained versions are also saved.

This means that if the free Colab session stopped before training finished, the project did not have to start again from the beginning.

The training code can resume from a previous checkpoint rather than starting completely from zero.

Training progress is also stored in JSON files.

---

# 19. Memory Optimization

The dataset is very large, while the free tier of Google Colab provides limited RAM and storage-related resources.

The Mel-spectrogram data alone can require several gigabytes.

A normal approach could try to load everything into RAM at once.

That caused memory problems and made it difficult to process and train on the full dataset within the available Colab resources.

The project was therefore modified to use **memory-mapped NumPy files**.

In simple terms:

Instead of:

> "Put the whole giant dataset on the table."

the system does:

> "Keep the giant dataset on the disk and bring me only the piece I need."

The dataset builder also writes features directly to disk rather than keeping the entire dataset in RAM.

During training, `tf.data.Dataset.from_generator()` is used so samples are loaded gradually.

These memory optimizations were implemented specifically to work within the resource limitations of the **free Google Colab tier**. They reduced RAM usage and made it possible to process the large dataset and train the models without requiring a paid Colab subscription or a more powerful local machine.

Together, checkpointing and memory optimization were important practical engineering solutions for completing the experiments under the limited and sometimes unstable resources provided by free Google Colab.

---

# 20. Evaluation

After training, the models were tested using the separate test dataset.

The evaluation code calculates:

### Accuracy

How many predictions were correct overall.

> "Out of everything, how often was the model correct?"

### Precision

When the model says:

> "This is fake."

how often is it actually fake?

### Recall

Out of all the actual fake recordings:

> "How many did the model successfully find?"

### Specificity

Out of all the actual real recordings:

> "How many did the model correctly recognize as real?"

### F1-score

A combined measure of precision and recall.

It is useful when the dataset is imbalanced.

### ROC-AUC

Measures how well the model separates real and fake across different decision thresholds.

Closer to 1 is generally better.

### EER

Equal Error Rate.

It is the point where the false-acceptance and false-rejection error rates are equal.

For this type of detection system:

> Lower EER is generally desirable.

The thesis defines these evaluation metrics as part of the evaluation framework.

---

# 21. Current Evaluation Results

The current test results are:

| Model          | Accuracy | Precision | Recall | Specificity |     F1 | ROC-AUC |    EER |
| -------------- | -------: | --------: | -----: | ----------: | -----: | ------: | -----: |
| CNN-only       |   97.09% |    98.71% | 98.07% |      87.69% | 98.39% |  99.06% |  4.94% |
| Hybrid CNN-RNN |   96.50% |    99.06% | 97.06% |      91.19% | 98.05% |  99.09% |  4.90% |
| RNN-only       |   86.03% |    97.29% | 87.00% |      76.82% | 91.86% |  91.06% | 17.45% |
| XGBoost        |   86.52% |    97.14% | 87.69% |      75.26% | 92.17% |  90.84% | 17.58% |
| AdaBoost       |   66.60% |    97.18% | 65.00% |      81.97% | 77.90% |  81.36% | 26.29% |

These values come from the current evaluation JSON files.

At this stage, these numbers are being **recorded as experimental results**. A detailed interpretation and discussion of why the models behaved this way will be performed separately.

---

# 22. Confusion Matrix Results

The confusion matrix gives us the exact number of correct and incorrect predictions.

The results are:

| Model          | True Negative | False Positive | False Negative | True Positive |
| -------------- | ------------: | -------------: | -------------: | ------------: |
| CNN-only       |         2,130 |            299 |            448 |        22,790 |
| Hybrid CNN-RNN |         2,215 |            214 |            684 |        22,554 |
| RNN-only       |         1,866 |            563 |          3,022 |        20,216 |
| XGBoost        |         1,828 |            601 |          2,860 |        20,378 |
| AdaBoost       |         1,991 |            438 |          8,134 |        15,104 |

The four terms mean:

### True Positive (TP)

The recording was fake and the model correctly said:

> Fake.

### True Negative (TN)

The recording was real and the model correctly said:

> Real.

### False Positive (FP)

The recording was real but the model incorrectly said:

> Fake.

### False Negative (FN)

The recording was fake but the model incorrectly said:

> Real.

---

# 23. Inference Speed

Inference means:

> **How long does the trained model need to make a prediction?**

The current neural-network evaluation reported:

| Model          |          Inference |
| -------------- | -----------------: |
| CNN-only       | 0.392 ms/sec audio |
| Hybrid CNN-RNN | 0.685 ms/sec audio |
| RNN-only       | 0.155 ms/sec audio |

These measurements were calculated by running prediction on the test set and dividing the elapsed prediction time by the total duration of the test audio.

The thesis also identifies inference time as an important part of evaluating whether the system can operate in real time.

---

# 24. Grad-CAM Explainability

The project also includes **Grad-CAM**.

Grad-CAM is basically a way of asking the CNN:

> **"Show me which part of the sound representation influenced your decision."**

The system creates a heatmap over the Mel-spectrogram.

For example, instead of only saying:

> "Fake: 98%"

the system can also show a visualization indicating which time-frequency areas contributed strongly to the prediction.

The implementation supports:

* Hybrid CNN-RNN
* CNN-only

because both contain a CNN/Mel-spectrogram branch.

It does not directly apply Grad-CAM to the RNN-only model because that model does not contain a convolutional layer.

The thesis identifies Grad-CAM/LRP as the planned explainability mechanism.

The current implementation uses **Grad-CAM**, not LRP.

---

# 25. Files and Results Saved

The project saves trained models, training progress, evaluation results and experiment information.

Because Google Colab can wipe or disconnect its temporary session storage when the runtime ends, files saved only inside the Colab environment may be lost. To prevent this, the project saves the important files to Google Drive.

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

The processed datasets, cached features, checkpoints and result files are also stored in Google Drive so they remain available after a Colab session ends or is restarted.

This makes it possible to resume training, reproduce experiments and inspect the results without immediately retraining every model.

---

# 26. What Has Been Successfully Completed

At this stage, the following major parts of the project have been implemented:

### Dataset

✓ ASVspoof 2019 LA processed
✓ Real/fake labels extracted from official protocols
✓ Approximately 121,461 files processed
✓ Speaker IDs used during splitting
✓ Speaker overlap check implemented

### Preprocessing

✓ 16 kHz sampling rate
✓ Mono audio
✓ Silence trimming
✓ Pre-emphasis
✓ Amplitude normalization
✓ Fixed 4-second duration

### Features

✓ 39-dimensional MFCC representation
✓ 128-band Mel-spectrogram
✓ 12-dimensional chroma representation
✓ Feature caching to disk

### Augmentation

✓ Gaussian noise
✓ Background noise support
✓ Speed perturbation
✓ Pitch shifting
✓ Compression-style augmentation
✓ Applied to training data only

### Deep Learning

✓ CNN-only model
✓ RNN-only model
✓ Hybrid CNN-RNN model
✓ CNN/RNN feature fusion
✓ Class weighting
✓ Adam optimizer
✓ Early stopping
✓ Best-model checkpointing
✓ Training resume support

### Traditional ML

✓ XGBoost baseline
✓ AdaBoost baseline
✓ Class imbalance handling

### Evaluation

✓ Accuracy
✓ Precision
✓ Recall
✓ Specificity
✓ F1-score
✓ ROC-AUC
✓ EER
✓ Confusion matrix
✓ Inference-time measurement

### Explainability

✓ Grad-CAM implemented
✓ CNN-only support
✓ Hybrid-model support
✓ Heatmap generation

---

# 27. What Has NOT Yet Been Fully Completed

Several items described in the thesis are either not yet implemented, or have not yet been demonstrated in the current experiments.

These should **not** be claimed as completed until they are actually implemented and tested.

Examples include:

* WaveFake cross-dataset testing
* VoiceWukong final evaluation
* AASIST2 comparison
* 1,000 bootstrap confidence intervals
* McNemar's statistical test
* Cohen's d effect-size analysis
* Full transfer-learning/adaptation experiments
* Full LRP explainability
* Production deployment/API
* Complete real-time application testing

The thesis proposes these components as part of the research methodology, but the current code/results supplied so far do not establish that all of them have been completed.

---

# 28. Current Project Status

The project has now reached an important milestone:

> **The complete basic experimental pipeline is working.**

The system can take the ASVspoof audio dataset, process the recordings, extract features, train multiple models, save the trained models, and evaluate them on a held-out test set.

The five main approaches have also been tested under the same evaluation framework.

This gives us the foundation needed for the next stage of the project:

**detailed analysis of the results.**

That next analysis should answer questions such as:

* Why did CNN perform so well?
* Why did the hybrid model have slightly different behavior?
* What does the difference between CNN and hybrid actually mean?
* Why did RNN perform much worse?
* Why did XGBoost behave similarly to RNN?
* Why did AdaBoost perform considerably worse?
* What do the false positives and false negatives tell us?
* Does the model satisfy the thesis requirements?
* Are the results genuinely strong, or could there be a methodological problem?
* Which thesis research questions can now be answered?
* Which experiments are still necessary before the project can be considered complete?

These questions should be analyzed separately from the documentation of what was implemented.
