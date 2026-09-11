# What this project actually does (explained without the jargon)

This is a quick rundown of the deepfake voice detection project for anyone
on the team who isn't deep into ML. If you already know what a CNN is, most
of this will be too basic for you — go read the thesis instead.

## The problem, in one paragraph

AI tools can now clone someone's voice from a few seconds of audio and make
it say anything. That's cool for accessibility apps, but it's also a fraud
goldmine — someone calls your bank pretending to be you, or a "CEO" leaves
a voicemail asking finance to wire money. Our job is to build something
that listens to a clip and tells you: was this a real human speaking, or
was it generated/cloned?

## Why this isn't as simple as "just use AI to catch AI"

The tricky part is that fake voices keep getting better. A detector trained
on last year's fake-voice tools might completely miss this year's tools.
So instead of chasing every new fake-audio generator individually, the
approach here is to look at the underlying *patterns* in how audio is put
together — spectral texture, pitch, rhythm — things that are genuinely hard
for a machine to fake perfectly, at least for now.

## The pipeline, step by step

Think of this as an assembly line. Raw audio goes in one end, a "real" or
"fake" label comes out the other end. Here's what happens in between.

**1. Cleaning up the audio**

Real-world audio recordings are messy — different volumes, silence at the
start/end, background hiss. Before the model can learn anything useful, we
run every clip through the same cleanup routine: resample it to a
consistent quality, trim the silence, boost the higher frequencies a bit
(this is a classic speech-processing trick that makes certain features
easier to pick out later), and normalize the volume so a whisper and a
shout end up on the same scale. Every clip also gets trimmed or padded to
exactly 4 seconds, so they're all the same length going into the model.

**2. Turning sound into numbers**

A neural network can't "listen" the way we do — it needs numbers. So we
convert each audio clip into a few different numerical representations:

- **MFCCs** — basically a compressed fingerprint of the sound's frequency
  content over time. This is the bread-and-butter feature in speech
  processing, been used for decades.
- **Mel-spectrograms** — a visual-ish representation of the audio, kind of
  like a heatmap of which frequencies are loud at which moments. You can
  actually treat this like an image and feed it to the kind of network
  normally used for photos.
- **Chroma features** — captures pitch-class information, useful for
  picking up on unnatural pitch patterns.

**3. Making the training data a bit noisier (on purpose)**

If you only ever train a model on pristine, studio-quality audio, it falls
apart the moment it hears a real phone call with background noise. So for
the training data only, we deliberately mess some of it up — add a bit of
noise, speed it up or slow it down slightly, shift the pitch a touch,
simulate compression artifacts. This is standard practice, it's called
"data augmentation" and it just means giving the model more variety to
learn from without needing to record more audio.

**4. Splitting the data properly**

This part matters more than it sounds like it should. If the same person's
voice shows up in both the training data and the test data, the model can
"cheat" — it learns to recognize that specific voice instead of learning
what makes fake audio sound fake in general. So we make sure every speaker
only ever appears in one of the three buckets: training, validation, or
testing. Roughly 60% of speakers for training, 20% for validation (used
during training to check progress), 20% held back for the final test.

**5. The actual model**

This is a "hybrid" model, meaning it's really two smaller models glued
together:

- One half (a CNN, the type of network usually used on images) looks at
  the Mel-spectrogram and picks up on texture-like patterns — the kind of
  visual artifacts that might show up when audio is synthetically
  generated.
- The other half (an RNN, specifically a bidirectional LSTM) looks at the
  MFCC sequence and pays attention to how things change over time — rhythm,
  pacing, the kind of temporal flow that's genuinely hard for a generator
  to nail perfectly.

Both halves' outputs get combined, and a final decision layer spits out a
probability: how likely is this real vs. fake.

**6. Comparing it against simpler methods**

We're also training a couple of more "old school" machine learning models
(XGBoost and AdaBoost, if you want the names) on the same data, just using
simplified statistics instead of the full sequences. This isn't because we
expect them to win — it's so we can actually show, with numbers, that the
fancier hybrid model is worth the extra complexity. If it turns out the
simple model does just as well, that's actually useful information too.

## Where things stand right now

- The full pipeline is written: cleanup, feature extraction, augmentation,
  splitting, the hybrid model, training, evaluation, and the two baseline
  comparisons.
- It hasn't been run on the real dataset yet (ASVspoof 2019 — the standard
  academic dataset for this kind of research, made up of real and
  AI-generated voice clips). Currently mid-download.
- Before that, we're doing a quick "smoke test" with a handful of random
  audio files just to make sure the code runs top to bottom without
  crashing. Getting good results isn't the point of that step — just
  confirming nothing's broken.
- A few things from the original thesis proposal got intentionally left
  out for now to keep this manageable as a student project — things like
  deploying it as a live cloud service, or comparing against some of the
  more exotic research models out there. Those are listed as "future work"
  rather than being skipped silently.

## How to actually run it (if you want to poke around)

Order matters:

1. `build_dataset.py` — reads the raw audio, cleans it, splits it, saves
   the processed features to disk.
2. `train.py` — trains the hybrid model on those features.
3. `evaluate.py` — runs the trained model against the held-out test set
   and prints out accuracy/precision/recall/etc.
4. `xgboost_baseline.py` / `adaboost_baseline.py` — same idea, but for the
   simpler comparison models.

Everything's configured in one place (`config.py`), so if you need to
change file paths, batch size, or training settings, that's the file to
touch.
