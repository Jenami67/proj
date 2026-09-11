# What Each File Does (Plain-Language Version)

Think of this whole project as a factory assembly line. Raw audio files go
in one end, and out the other end comes a verdict: "real voice" or "fake
voice." Each file below is one station on that assembly line, or a tool
used by one of those stations.

---

## `config.py` — The settings sheet

This is just a list of numbers and folder paths that every other file
needs to agree on — things like "audio should be 16,000 samples per
second" or "cut every clip down to 4 seconds." Instead of typing those
numbers separately into every file (and risking a typo somewhere), every
file reads them from here. Change a number once in `config.py`, and it
changes everywhere automatically.

## `protocols.py` — The answer key reader

The ASVspoof dataset comes with special text files (called "protocol
files") that work like an answer key: for every audio filename, they say
whether that recording is a real human voice or an AI-generated fake, and
which person's voice it is. This file's job is just to read that answer
key and turn it into a simple list your computer can use: "this file is
real, spoken by person #79" and so on.

## `audio_utils.py` — The cleanup station

Raw audio recordings are messy — different volumes, silence at the start
and end, background hiss. This file cleans every clip up the same way,
every time, so the computer is comparing apples to apples:
1. Load the audio and make sure it's the same "resolution" (sample rate)
2. Trim off the silent parts at the start/end
3. Slightly boost the higher frequencies (a standard trick in speech
   processing that makes certain voice characteristics easier to detect)
4. Turn the volume up or down so every clip peaks at the same loudness
5. Cut or pad every clip to exactly the same length (4 seconds)

## `augmentation.py` — The "make it harder" station

This one only touches the **training** audio, never the val/test audio.
It deliberately messes up copies of the training clips — adding noise,
speeding them up or down slightly, changing pitch a little, roughening
the audio like it went through a bad phone call. Why deliberately damage
your own training data? Because it teaches the model not to get fooled by
messy, real-world audio conditions later — like a student who practices
with distractions in the room so a quiet exam room feels easy by
comparison.

## `features.py` — The translator

Neural networks can't "listen" to a raw sound wave in any useful way — it
needs to be turned into a picture-like format first. This file does that
translation, producing three different "pictures" of the same audio:
- **MFCC** — a compact summary of the voice's tone/timbre over time
- **Mel-spectrogram** — a detailed picture of which frequencies are loud
  at which moments (literally looks like a colorful heatmap image)
- **Chroma** — a summary of which musical pitch classes are present

## `build_dataset.py` — The factory manager

This is the file that actually runs the whole preparation process. It:
1. Asks `protocols.py` for the full list of files and their real/fake
   labels
2. Splits everyone into three groups — training, validation, and
   testing — making sure the **same person's voice never appears in two
   different groups** (otherwise the model could "cheat" by recognizing a
   specific voice instead of learning what makes speech fake in general)
3. Sends training audio through `augmentation.py`'s "make it harder"
   station (val/test audio skips this step — you never want to trick your
   own final exam)
4. Sends everything through `audio_utils.py`'s cleanup and then
   `features.py`'s translator
5. Saves the final results to disk so you don't have to redo all this
   work every time you want to train a model

## `prepare_test_data.py` — The practice dummy

This is a temporary helper — not a permanent part of the project. Since
you don't have the real dataset yet, this file fakes a tiny 10-file
version of it (with a made-up answer key), just so you can run the whole
assembly line once and confirm nothing breaks. Delete it once the real
dataset is downloaded.

## `model.py` — The blueprint for the brain

This is where the actual detection model is designed — not trained yet,
just the blueprint of how information flows through it. It has two
"brains" working together:
- A **CNN brain** that looks at the Mel-spectrogram picture and learns to
  spot visual patterns/textures that are typical of fake speech
- An **RNN brain** that reads the MFCC data in order, like reading a
  sentence word by word, and learns to notice timing patterns real voices
  have that fakes often miss

Both brains' conclusions get combined before the model makes its final
real/fake decision. Two simpler "one-brain-only" versions are also
defined here, just for comparison later — to prove the combined approach
is actually worth the extra complexity.

## `train.py` — The classroom

This is where the model actually learns. It shows the model the training
examples over and over (an "epoch" = one full pass through all of them),
checks itself against the validation set after each pass, and gradually
adjusts itself to get better at telling real from fake. It automatically
stops early if it stops improving, and saves the best version it ever
reached.

## `metrics_utils.py` — The report card generator

Once a model has made its predictions, this file scores how well it did —
not just "% correct," but several different ways of measuring correctness
that matter for this kind of problem (e.g., how often it wrongly accuses
a real voice of being fake, vs. how often it lets a fake voice through).

## `evaluate.py` — The final exam

This runs the trained model against audio it has never seen before (the
test set) and calls `metrics_utils.py` to produce the report card. This
is the honest, final measurement of how good the model actually is.

## `baseline_utils.py`, `xgboost_baseline.py`, `adaboost_baseline.py` — The "old-school" competitors

These three files build two much simpler, non-neural-network models
(XGBoost and AdaBoost — classic machine learning techniques from before
deep learning became popular) using the exact same audio features. The
point isn't that these are expected to win — it's to prove that the
fancy CNN+RNN model is actually earning its complexity by beating these
simpler, faster alternatives. If it doesn't, that's an important, honest
finding too.

---

### The order things actually run in

```
1. (once, temporarily)  prepare_test_data.py     -- fakes a tiny dataset
2. build_dataset.py                              -- cleans + prepares everything
3. train.py                                      -- teaches the model
4. evaluate.py                                   -- grades the model
5. xgboost_baseline.py / adaboost_baseline.py     -- grades the "old-school" competitors
```
