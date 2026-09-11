# What Each File Does (Technical Version)

Maps each file to the thesis, states the concrete design decisions made,
and flags anything you should keep in mind when extending it.

---

## `config.py`
Single source of truth for every constant used across the pipeline —
paths, audio params, feature dims, split ratios, augmentation
probabilities, training hyperparameters. Everything else imports from
here; nothing hardcodes these values elsewhere.

- `LA_ROOT` and the `TRAIN/DEV/EVAL_FLAC_DIR` + `*_PROTOCOL` paths assume
  the **unmodified official ASVspoof2019 LA folder structure** — just
  extract the official zip into `dataset/LA/` and these paths resolve
  automatically. No manual re-sorting into `fake/`/`real/` folders needed
  anymore (that was the earlier, less faithful approach).
- `ALL_PARTITIONS` is consumed by `build_dataset.py`'s `gather_files()` —
  train+dev+eval get pooled and then re-split ourselves (see below).
- `N_MFCC = 13` — this is the *base* coefficient count; `features.py`
  adds delta + delta-delta on top to reach the 39-dim total FUN-03 asks
  for. Don't confuse this constant with the final feature dimensionality.
- `DURATION_SECONDS = 4` — deliberate simplification of Table 3's
  ambiguous "4s if shorter / central 6s if longer" rule. Documented
  inline; revisit only if you decide the 6s branch matters for your
  results.
- `BATCH_SIZE = 16` — kept conservative since your GPU/CPU situation
  isn't confirmed. Bump to 32+ once you know your hardware (see the open
  question from earlier).

## `protocols.py`
Parses ASVspoof's CM (countermeasure) protocol file format directly —
this is the actual, correct way to get ground-truth labels and speaker
IDs for ASVspoof, replacing the earlier filename-guessing heuristic that
would have silently failed on the real dataset.

- `parse_protocol()`: reads whitespace-delimited lines
  (`SPEAKER_ID FILENAME - SYSTEM_ID KEY`), maps `bonafide`→0,
  `spoof`→1. Any line with fewer than 5 fields is silently skipped
  (guards against blank trailing lines).
- `build_file_list()`: joins protocol entries against what's actually
  present on disk, so a corrupted/partial download doesn't crash the
  whole pipeline — it just reports how many files were skipped and
  continues with what's available. Returns parallel lists (not a
  DataFrame) to keep this dependency-light and match the rest of the
  codebase's style.

## `audio_utils.py`
Implements the fixed preprocessing chain from Table 3, **in table order**:
resample → silence trim → pre-emphasis → peak normalize → duration fix.

- `trim_silence()`: `librosa.effects.trim()` is dB/amplitude-threshold
  based, not true energy-VAD — the thesis's requested "energy threshold
  0.01×max amplitude" is converted into an equivalent `top_db` value
  (`-20*log10(0.01) = 40 dB`). This is the standard approximation when
  librosa is your toolset; a true VAD would need `webrtcvad` or similar.
- `normalize_audio()` runs **last**, after `pre_emphasis()` — this was a
  real bug in the first version (normalizing before pre-emphasis doesn't
  guarantee the post-filter signal stays in [-1, 1], since the filter
  changes amplitude scale). Fixed now; don't reorder these two calls in
  `preprocess_audio()` without re-checking this.
- `standardize_duration()`: center-crop if too long, zero-pad (trailing)
  if too short. This is what guarantees every sample produces
  identically-shaped MFCC/Mel/chroma arrays downstream, which is required
  for `np.array(list_of_arrays)` to work without ragged-array errors.

## `augmentation.py`
Section 3.1.5 Step 3, applied **only** where `build_dataset.py` passes
`augment=True` (training split only).

- Percentages from the thesis (30/20/30/20/15%) are read as **per-sample
  independent probabilities**, not fixed dataset subsets — this is both
  the more standard interpretation and the simpler one to implement.
- Implemented as a one-time **offline** step (computed once when
  `build_dataset.py` runs, cached to disk) rather than true epoch-by-epoch
  **online** augmentation as the thesis's wording technically implies.
  True online augmentation would mean wrapping librosa calls in
  `tf.py_function` inside a `tf.data` pipeline so fresh random distortions
  get applied every epoch — doable, but real added complexity for
  comparatively little practical benefit at this project's scale. If
  you want to revisit this later (e.g. for a stronger overfitting
  argument in your results chapter), flag it and we can build it.
- `add_background_noise()` no-ops gracefully if `dataset/background_noise/`
  doesn't exist or is empty — the thesis doesn't specify a noise source
  dataset, so this is left as an optional drop-in.
- `mp3_compression_emulation()` adds quantization-style noise rather than
  doing a real MP3 encode/decode round-trip (which would need `pydub` +
  a system `ffmpeg` install) — a standard, dependency-light substitute.
- Every augmentation function re-runs `standardize_duration()` +
  `normalize_audio()` at the end of `apply_augmentations()`, since
  `speed_perturb`/`pitch_shift` change sample count and amplitude.

## `features.py`
FUN-03/FUN-04. Three extractors, one entry point (`extract_all_features`).

- `extract_mfcc()`: `librosa.feature.mfcc(n_mfcc=13)` + `librosa.feature.delta()`
  order 1 and order 2, concatenated along the coefficient axis → shape
  `(39, time)`, transposed to `(time, 39)` for Keras's expected
  `(timesteps, features)` convention for RNN input.
- `extract_mel_spectrogram()`: 128 mel bands, converted to log scale via
  `librosa.power_to_db(ref=np.max)` — feeding raw linear power to a CNN
  is nonstandard; log-mel is the near-universal convention in speech deep
  learning.
- `extract_chroma()`: standard 12-bin `chroma_stft`.
- All three share `N_FFT=2048`, `HOP_LENGTH=512` from `config.py`, which
  is why every feature type ends up with the same `time` dimension for a
  given fixed-duration clip — important, since `model.py`'s CNN and RNN
  branches are fed from the same batch and don't need to be re-aligned.

## `build_dataset.py`
Orchestrator — this is the file you actually run first.

- `gather_files()`: pools **train + dev + eval** from ASVspoof's official
  protocols into a single flat list, rather than using ASVspoof's own
  partitions as your train/val/test directly. This is a deliberate
  choice: ASVspoof's official eval set is designed for the ASVspoof
  *challenge* (unseen attack types, adversarial by design); pooling
  everything and doing your own speaker-independent 60/20/20 split
  (Section 3.1.4) gives you a split ratio and process that actually
  matches what your methodology chapter describes. Worth stating
  explicitly in your report so it doesn't read as accidental.
- `speaker_aware_split()`: two chained `GroupShuffleSplit` calls
  (60% train, then 50/50 of the remainder for val/test), grouped by
  speaker ID from the protocol file — real speaker independence now,
  not the filename-heuristic fallback from before.
- There's a hard **`assert`** right after the split that checks zero
  speaker overlap across train/val/test and fails loudly if that's ever
  violated — treat this as a correctness guarantee, not just a print
  statement.
- `build_features_for_split()` is where `audio_utils` → (optionally)
  `augmentation` → `features` actually get called per file, per split.
  Progress is logged every 200 files (adjust if your dataset is much
  smaller/larger).
- Output: 4 `.npy` files per split (`mfcc`, `mel`, `chroma`, `y`) in
  `data/splits/` — everything downstream (`train.py`, `evaluate.py`, both
  baselines) reads from these instead of ever touching raw audio again.

## `prepare_test_data.py`
Not part of the real pipeline — a throwaway smoke-test generator.

- Takes any 10 `.wav` files you point it at, fakes up an ASVspoof-shaped
  folder + protocol structure so `build_dataset.py` can run against it
  unmodified. Copies your `.wav` files under `.flac` names — works
  because `soundfile`/`librosa` sniff the file header, not the extension,
  but flag it to me if you ever see a load error, since that's the first
  place I'd look.
- **Known limitation, not a bug:** with only 5 fake speaker groups across
  10 files, the speaker-aware split can (and did, in your run) produce a
  test split with only one class present. `metrics_utils.py` now handles
  that gracefully (`roc_auc`/`eer` report as `None` with a warning printed)
  instead of crashing. This tells you nothing about real dataset
  performance — its only job is proving the plumbing runs without
  exceptions.
- Delete this file once `dataset/LA/` has the real ASVspoof2019 LA data.

## `model.py`
Section 3.2.1 / Figure 2 — two-branch hybrid architecture, functional
Keras API (not Sequential, since it needs two separate inputs).

- `build_cnn_branch()`: takes the Mel-spectrogram `(time, 128)`, reshapes
  to `(time, 128, 1)` to add a channel dim for `Conv2D`, three
  Conv+BatchNorm+Pool blocks (32→64→128 filters), ends in
  `GlobalAveragePooling2D()` instead of `Flatten()` — keeps the parameter
  count much lower and is the standard modern choice over flattening.
- `build_rnn_branch()`: takes the MFCC sequence `(time, 39)`, two stacked
  `Bidirectional(LSTM)` layers (64 units each), first returns sequences
  for the second to consume, second collapses to a single vector.
- `build_hybrid_model()`: `Concatenate` layer fuses both branches' output
  vectors, then Dense(128)→Dropout(0.3)→Dense(64)→Dropout(0.3)→
  Dense(2, softmax). Dropout rate is a reasonable default, not
  thesis-specified — tune if you see over/underfitting once real
  training happens.
- `build_cnn_only_model()` / `build_rnn_only_model()`: single-branch
  variants sharing the same branch-building functions, needed to answer
  Research Question 3 (hybrid vs. single-architecture comparison).
- Chroma features are extracted (`features.py`) but **not yet wired into
  this model** — currently only feeds the deep learning classical
  baselines (via `baseline_utils.py`). If you want a three-branch fusion
  including chroma in the neural model itself, that's a straightforward
  extension of this file — say the word.

## `train.py`
Section 3.1.6 training config, applied literally:

- `Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)` — thesis-specified
  values, not Keras defaults (Keras default beta_2 is the same, but
  stating both explicitly for clarity/reproducibility).
- `categorical_crossentropy` + `compute_class_weight("balanced", ...)` —
  implements "loss function is categorical cross-entropy with class
  weighting" directly. Labels are one-hot encoded via
  `tf.keras.utils.to_categorical` since the model's final layer is
  `softmax` over 2 units, not a single sigmoid unit.
- `EarlyStopping(patience=8, restore_best_weights=True)` +
  `ModelCheckpoint(save_best_only=True)` — standard training hygiene, not
  explicitly demanded by the thesis but consistent with its emphasis on
  avoiding overfitting; also directly useful for producing the
  "Preliminary Findings"-style table your methodology chapter references.
- Two model files get saved: `hybrid_cnn_rnn_best.keras` (best validation
  loss, via checkpoint) and `hybrid_cnn_rnn_final.keras` (whatever weights
  training ended on, which with `restore_best_weights=True` should
  usually match the best checkpoint anyway — kept both just in case).

## `metrics_utils.py`
Table 4, factored out into a shared module so `evaluate.py` and both
baseline scripts compute identical metrics identically — no risk of the
hybrid model and the XGBoost baseline being scored with subtly different
logic.

- `compute_eer()`: EER isn't in scikit-learn, so it's derived manually
  from `roc_curve()` — find the threshold where false-positive rate and
  false-negative rate are closest, average them. Matches the standard
  definition used in anti-spoofing literature (Todisco et al., 2019).
  Returns `None` (not a crash, not a fake `0.0`) if `y_true` only has one
  class — mathematically EER isn't defined in that case, and silently
  returning 0 would be misleading.
- `compute_all_metrics()` centralizes the single-class guard and prints a
  warning if triggered, so it's visible in the console output rather than
  silently producing `None` with no explanation.

## `evaluate.py`
Thin driver: loads the saved test split, loads a trained `.keras` model,
predicts, times the prediction (`inference_ms_per_sec_audio` — this is
your NON-FUN-01/Table 4 "inference time" metric), and calls
`metrics_utils` to score it. Deliberately has almost no logic of its own
— everything reusable lives in `metrics_utils.py` and `model.py` so this
file stays a thin, easy-to-read entry point.

## `baseline_utils.py`
Bridges the deep-learning feature format (`(n_samples, time, n_dims)`
sequences) to what classical ML models expect (`(n_samples, n_dims)` flat
vectors).

- `summarize_over_time()`: mean + std of each feature dimension across
  the time axis — a simplified stand-in for Bird & Lotfi (2023)'s
  temporal feature engineering. Doubles the feature count (mean and std
  per dimension) but keeps this a completely standard, well-understood
  aggregation rather than something exotic.
- `load_flat_features()`: concatenates the flattened MFCC + Mel + chroma
  vectors into one long feature vector per sample — this is what actually
  goes into XGBoost/AdaBoost.

## `xgboost_baseline.py` / `adaboost_baseline.py`
Section 3.1.6's comparative baselines, using `baseline_utils` for feature
prep and `metrics_utils` for scoring — same evaluation code path as the
main model, so results are directly comparable in your results chapter.

- XGBoost: `n_estimators=200, max_depth=6, learning_rate=0.1` — reasonable
  defaults for a first pass, not exhaustively tuned; if XGBoost
  suspiciously underperforms, hyperparameter tuning here is the first
  thing to check before concluding the hybrid model is definitively
  better.
- AdaBoost: thesis's baseline (Nair et al., 2024) used AdaBoost with
  **pause-based biological features**, which aren't implemented here
  (documented earlier as an optional extension — needs `parselmouth` or
  similar beyond Librosa). This AdaBoost baseline currently runs on the
  same mean/std spectral features as XGBoost, so it's a fair
  architecture-vs-architecture comparison, but **not** a reproduction of
  Nair et al.'s actual method — say so explicitly if you report this
  number against their published 79-81% accuracy, since it's not an
  apples-to-apples comparison.

---

## Run order

```bash
python prepare_test_data.py /path/to/some/wav/files   # smoke test only, once
python build_dataset.py                                # real pipeline entry point
python train.py
python evaluate.py
python xgboost_baseline.py
python adaboost_baseline.py
```
