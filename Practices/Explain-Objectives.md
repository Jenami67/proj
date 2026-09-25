# How We Achieved the Research Objectives

This document explains, in very simple terms, **what we actually did to achieve each research objective in the thesis**.

This document connects everything directly to the **six research objectives in Section 1.4 of the thesis**.

The basic idea is:

> **Objective → What we needed to do → What we actually did → Which files did the work**

---

# Objective 1

## Identify and acquire suitable public datasets

### What the thesis wanted

The thesis proposed using public datasets containing:

* Real human speech
* AI-generated/synthetic speech

### What we did

We used **ASVspoof 2019 Logical Access (LA)** as the main dataset for the implemented system.

It contains both:

* Real speech
* Spoofed/fake speech

We obtained the dataset and used its official protocol information to determine which audio files were real and which were fake.

We created separate training, validation and testing data while keeping the speaker separation required for the experiment.

### Files responsible for this objective

**`protocols.py`**

Used to work with the ASVspoof protocol files (.txt files) and determine the labels of the recordings.

**`build_dataset.py`**

Used to actually build the usable dataset from the original ASVspoof files to make our train.py use the wav files.

**`prepare_test_data.py`**

Used for preparing test/evaluation data.

**`config.py`**

Contains important dataset and project configurations like global variables to use in other .py files.

### More explanation

Think of ASVspoof as a giant box containing thousands of audio recordings.

We needed to tell the computer:

> "This recording is REAL."

or:

> "This recording is FAKE."

`protocols.py` read the .txt files in the dataset and helped identify the labels, and `build_dataset.py` prepared everything for the AI.

---

# Objective 2

## Preprocess audio and extract MFCC, Mel-spectrogram and chroma features from them

### What the thesis wanted

The thesis wanted the raw audio to be cleaned and converted into useful representations.

Instead of giving the AI a raw WAV file and saying:

> "Figure this out."

we first prepared the audio and extracted useful information from it.

The main features were:

* MFCC
* Mel-spectrogram
* Chroma

### What we did

Our processing pipeline was:

```text
Audio
  ↓
Preprocessing
  ↓
Augmentation
  ↓
Feature Extraction
  ↓
Saved Features
```

The audio was processed into a standard format and then augmented during dataset preparation to make prediction in real life better.

After that, the features were extracted.

### MFCC

We extracted:

**13 MFCC coefficients**

and also calculated:

* Delta
* Delta-delta

This resulted in:

**39 MFCC-related features**

### Mel-spectrogram

We extracted a:

**128-band Mel-spectrogram**

This gives the neural network a picture-like representation of the audio's frequency content over time.

### Chroma

We also extracted:

**12 chroma features**

These provide information related to the pitch-class content of the audio.

### Files responsible for this objective

**`audio_utils.py`**

Handles the main audio preprocessing operations.

**`augmentation.py`**

Handles the audio augmentation operations.

**`features.py`**

Handles feature extraction such as:

* MFCC
* Mel-spectrogram
* Chroma

**`build_dataset.py`**

Connects these steps together and creates the final processed dataset.

### More explanation

Imagine giving the AI a messy audio recording.

First we clean it.

Then we slightly modify it for training.

Then we turn the audio into useful "pictures/numbers":

```text
MFCC       → speech characteristics
Mel        → frequency/spectral patterns
Chroma     → pitch-related information
```

The AI does not directly eat the original WAV file during training.

It eats these prepared features.

---

# Objective 3

## Design a hybrid CNN-RNN architecture

### What the thesis wanted

The main model was supposed to combine:

**CNN + RNN**

The idea was simple:

### CNN

The CNN looks for patterns in the spectral representation.

For example:

> "Does this sound contain an unusual frequency pattern?"

### RNN/LSTM

The RNN looks at how the audio information changes over time.

For example:

> "Does something unusual happen in the sequence of the speech?"

The two approaches are then combined.

### What we did

We implemented three neural-network models so that we could compare them:

```text
CNN-only
RNN-only
Hybrid CNN-RNN
```

The main proposed model was the:

**Hybrid CNN-RNN**

The hybrid model uses:

```text
Mel-spectrogram
      ↓
     CNN
      ↓
   Features
      ↘
       Fusion → Classification → Real/Fake
      ↗
MFCC → RNN/LSTM
```

So the model gets two different views of the same audio:

* CNN → spectral patterns
* RNN/LSTM → temporal patterns

Then it combines them before making the final decision.

### Files responsible for this objective

**`model.py`**

Contains the neural-network model definitions.

This is where the CNN-only, RNN-only and hybrid architectures are implemented.

**`config.py`**

Contains model/training configuration.

### More explanation

Imagine two people listening to the same audio.

Person #1 is good at spotting **patterns in the sound picture**.

That's the CNN.

Person #2 is good at remembering **what happened before and what happens next**.

That's the RNN/LSTM.

The hybrid model lets both people work together.

---

# Objective 4

## Train the proposed model and optimize it

### What the thesis wanted

After building the model, we needed to teach it how to distinguish:

```text
REAL
```

from:

```text
FAKE
```

The thesis also specified training settings such as:

* Adam optimizer
* Learning rate
* Class weighting
* Training/validation data
* Early stopping
* Hyperparameter configuration

### What we did

We trained:

* CNN-only
* RNN-only
* Hybrid CNN-RNN

We also used class weighting because the dataset contains **far more fake recordings than real recordings**.

Without handling this imbalance, the model could become biased toward the majority class.

We also used:

* Training data
* Validation data
* Checkpoints
* Early stopping
* Batch-based training

Training was performed in Google Colab.

### Main training command

The main training process was started with:

```bash
!python train.py
```

### Files responsible for this objective

**`train.py`**

This is the main training program and most time and resource consuming one.

It handles:

* Loading the prepared data
* Creating training batches
* Training the neural networks
* Validation
* Class weighting
* Checkpoints
* Early stopping

**`model.py`**

Defines the models that `train.py` trains.

**`config.py`**

Contains training/model settings.

**`train_baselines.py`**

train other models for the traditional baseline comparision.

### More explanation

`model.py` says:

> "This is what the brain looks like."

`train.py` says:

> "Now teach this brain using thousands of examples (audio files)."

The model sees examples of:

```text
REAL → correct answer: REAL
FAKE → correct answer: FAKE
```

and gradually adjusts itself so that it becomes better at classification. Like a baby learning animals pictures.

---

# Objective 5

## Evaluate the model and compare it with other approaches

### What the thesis wanted

After training, we could not simply say:

> "The model looks good."

We needed numbers.

The thesis specifies metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The project also calculated additional useful metrics such as:

* Specificity
* EER
* Confusion matrix
* Inference time

### What we did

We evaluated the trained models on data that was not used to train them (the val part of dataset splits).

We tested:

```text
Hybrid CNN-RNN
CNN-only
RNN-only
XGBoost
AdaBoost
```

This gave us a meaningful comparison between:

**Deep learning approaches**

and

**Traditional machine-learning approaches**

### Neural-network evaluation

We used:

```bash
!python evaluate.py --model hybrid
```

for the hybrid model.

```bash
!python evaluate.py --model cnn_only
```

for the CNN-only model.

And:

```bash
!python evaluate.py --model rnn_only
```

for the RNN-only model.

### Traditional ML comparison

We also ran:

```bash
!python xgboost_baseline.py
```

and:

```bash
!python adaboost_baseline.py
```

### Files responsible for this objective

**`evaluate.py`**

Evaluates the trained neural-network models.

**`metrics_utils.py`**

Contains the calculation of evaluation metrics.

**`xgboost_baseline.py`**

Implements the XGBoost comparison model.

**`adaboost_baseline.py`**

Implements the AdaBoost comparison model.

**`baseline_utils.py`**

Provides common functionality for the baseline models.

**`statistical_analysis.py`**

Contains statistical-analysis functionality included in the project.

### More explanation

We trained several "students":

```text
CNN
RNN
CNN + RNN
XGBoost
AdaBoost
```

Then we gave all of them an exam they had not seen before.

The evaluation scripts calculated the scores.

That lets us answer:

> "How well does our model actually work?"

instead of just saying:

> "It seems good."

---

# Objective 6

## Develop a real-time or near-real-time detection framework

### What the thesis wanted

The final system should not only exist as training code.

A user should be able to give it an audio file and receive a result such as:

```text
REAL
```

or:

```text
FAKE
```

The thesis also aimed for fast inference suitable for practical applications.

### What we did

We created a user interface using **Streamlit**.

The main application file is:

**`app.py`**

The application allows the user to:

1. Open the web interface.
2. Upload a WAV audio file.
3. Process the audio.
4. Extract the required features.
5. Send the features to the trained model.
6. Receive a prediction.
7. Display whether the audio is **Real or Fake**.

The user therefore does not need to run Python commands manually.

They can simply use the application.

### Main application file

**`app.py`**

This is the front-end application.

It connects the user interface to the trained detection system.

Other project files provide the processing behind it.

For example:

```text
app.py
   ↓
audio processing
   ↓
feature extraction
   ↓
trained model
   ↓
prediction
   ↓
Real / Fake
   ↓
Confidence percent 
    ↓
Show Grad-Cam images
 and Descriptions
```

This demonstrates that the trained models can perform inference very quickly in the tested environment. 

### Files responsible for this objective

**`app.py`**

Provides the Streamlit interface and prediction workflow.

**`audio_utils.py`**

Processes uploaded audio.

**`features.py`**

Extracts the required features.

**`model.py`**

Provides the model architecture.

**Saved model files in models folder**

Provide the trained model used for prediction.

### More explanation

Before:

> We had a trained brain sitting on a computer.

After:

> We gave the brain a simple website.

The user uploads:

```text
voice.wav
```

The system processes it and returns:

```text
REAL
```

or:

```text
FAKE
```

So the project moved from **"AI model in a notebook"** toward **"something a person can actually use."**

---

# Final Mapping: Objective → What We Built

| Objective                       | What we achieved                                                                                                | Main files                                                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **1. Dataset**                  | Acquired and processed ASVspoof 2019 LA with real/fake labels                                                   | `protocols.py`, `build_dataset.py`, `prepare_test_data.py`, `config.py`                                                          |
| **2. Preprocessing & features** | Preprocessed audio and extracted MFCC, Mel and Chroma features                                                  | `audio_utils.py`, `augmentation.py`, `features.py`, `build_dataset.py`                                                           |
| **3. Hybrid architecture**      | Built CNN-only, RNN-only and Hybrid CNN-RNN models                                                              | `model.py`, `config.py`                                                                                                          |
| **4. Training & optimization**  | Trained models with validation, class weighting, early stopping and checkpoints                                 | `train.py`, `model.py`, `config.py`                                                                                              |
| **5. Evaluation & comparison**  | Evaluated CNN, RNN, Hybrid, XGBoost and AdaBoost using multiple metrics                                         | `evaluate.py`, `metrics_utils.py`, `xgboost_baseline.py`, `adaboost_baseline.py`, `baseline_utils.py`, `statistical_analysis.py` |
| **6. Detection framework**      | Built a Streamlit application for uploading audio and receiving Real/Fake predictions; measured inference speed | `app.py`, `audio_utils.py`, `features.py`, `model.py`                                                                            |

---

# What We Have Actually Completed

Putting the six objectives together, our implemented system now looks like this:

```text
                 ASVspoof 2019 LA
                        ↓
                Dataset Preparation
                        ↓
                    Splitting
                        ↓
                 Audio Preprocessing
                        ↓
                    Augmentation
                        ↓
                Feature Extraction
             ↙          ↓          ↘
           MFCC        Mel        Chroma
             ↓          ↓
             ↓          ↓
          RNN/LSTM     CNN
             ↘          ↙
              Hybrid CNN-RNN
                      ↓
                 Real / Fake
                      ↓
                  Evaluation
                      ↓
              Performance Results
                      ↓
                Streamlit App
                      ↓
               Upload WAV → Result
```

The additional comparison models fit beside the main model:

```text
                 Prepared Features
                        ↓
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
        CNN-only      RNN-only    Hybrid CNN-RNN
          ↓             ↓             ↓
          └─────────────┼─────────────┘
                        ↓
                  Compare Results
                        ↑
                   XGBoost
                   AdaBoost
```

# One-Sentence Version for Each Objective

If the monkey needs to explain the thesis defense very quickly:

**Objective 1:**

> We acquired ASVspoof 2019 LA and prepared its real and fake recordings for the experiment.

**Objective 2:**

> We cleaned and standardized the audio, applied augmentation, and extracted MFCC, Mel-spectrogram and chroma features.

**Objective 3:**

> We built CNN, RNN and the main Hybrid CNN-RNN architecture so spectral and temporal information could be analyzed together.

**Objective 4:**

> We trained the models using the prepared features with class weighting, validation, early stopping and checkpointing.

**Objective 5:**

> We evaluated the models using accuracy, precision, recall, F1, specificity, ROC-AUC, EER and confusion matrices, and compared them with XGBoost and AdaBoost.

**Objective 6:**

> We built a Streamlit application that allows a user to upload a WAV file and receive a Real/Fake prediction, and we measured the model's inference speed.

#
