"""Evaluation (Table 4): accuracy, precision, recall, specificity, F1,
ROC-AUC, EER, confusion matrix, and inference time.

This is a thin driver -- all metric math lives in metrics_utils.py so the
numbers are computed identically here and in the baseline scripts. Earlier
versions of this file reimplemented compute_eer() itself, which meant a fix
in one place didn't propagate to the other, and it didn't handle the
single-class-in-y_true edge case (metrics_utils.py does).

Usage:
    python evaluate.py                    # evaluates the hybrid model
    python evaluate.py --model cnn_only    # evaluates CNN-only baseline
    python evaluate.py --model rnn_only    # evaluates RNN-only baseline
    python evaluate.py --model-path some/path.keras
"""

import argparse
import json
import time

import numpy as np
import tensorflow as tf

import config
from metrics_utils import compute_all_metrics, print_metrics

# Maps a short --model name to its saved-checkpoint filename and to which
# feature arrays it needs as input. Keeps evaluate.py usable for the
# hybrid model and both single-branch baselines without hardcoding paths.
MODEL_REGISTRY = {
    "hybrid": {
        "filename": "hybrid_cnn_rnn_best.keras",
        "inputs": "both",
    },
    "cnn_only": {
        "filename": "cnn_only_best.keras",
        "inputs": "mel",
    },
    "rnn_only": {
        "filename": "rnn_only_best.keras",
        "inputs": "mfcc",
    },
}


def load_test_split():
    mel = np.load(config.SPLITS_DIR / "test_mel.npy")
    mfcc = np.load(config.SPLITS_DIR / "test_mfcc.npy")
    y = np.load(config.SPLITS_DIR / "test_y.npy")
    return mel, mfcc, y


def build_model_inputs(mel, mfcc, inputs_kind):
    """Different model kinds expect different input shapes: the hybrid
    model takes [mel, mfcc], the CNN-only baseline takes just mel, and the
    RNN-only baseline takes just mfcc."""
    if inputs_kind == "both":
        return [mel, mfcc]
    elif inputs_kind == "mel":
        return mel
    elif inputs_kind == "mfcc":
        return mfcc
    raise ValueError(f"Unknown inputs_kind: {inputs_kind}")


def evaluate_model(model_path, inputs_kind="both"):
    mel, mfcc, y_true = load_test_split()
    model = tf.keras.models.load_model(model_path)
    model_inputs = build_model_inputs(mel, mfcc, inputs_kind)

    start = time.time()
    probs = model.predict(model_inputs, batch_size=config.BATCH_SIZE)
    elapsed = time.time() - start

    y_pred = np.argmax(probs, axis=1)
    y_score_fake = probs[:, 1]  # probability of the "fake" class

    total_audio_seconds = len(y_true) * config.DURATION_SECONDS
    ms_per_second_audio = (
        (elapsed * 1000) / total_audio_seconds if total_audio_seconds else 0
    )

    results = compute_all_metrics(y_true, y_pred, y_score_fake)
    results["inference_ms_per_sec_audio"] = ms_per_second_audio
    return results


def save_results(results, model_name):
    """Writes results to results/<model_name>_eval.json so Table 7-style
    numbers can be pulled into the thesis without re-running evaluation."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"{model_name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="hybrid",
        help="Which registered model to evaluate (default: hybrid).",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Optional explicit path to a .keras file, overriding --model.",
    )
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
        inputs_kind = "both"  # assume hybrid-style inputs unless told otherwise
        model_name = "custom"
    else:
        entry = MODEL_REGISTRY[args.model]
        model_path = config.MODELS_DIR / entry["filename"]
        inputs_kind = entry["inputs"]
        model_name = args.model

    results = evaluate_model(model_path, inputs_kind)
    print_metrics(results, title=f"Evaluation results ({model_name}):")
    save_results(results, model_name)


if __name__ == "__main__":
    main()
