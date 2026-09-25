"""AdaBoost baseline (Section 3.1.6 comparative baselines; Nair et al.
2024 used AdaBoost with pause-based features).

We don't implement the pause-based features here (jitter/shimmer/pause
timing require extra libraries beyond Librosa -- flagged as a separate,
optional extension in the project notes, not needed for a first working
version). As a simpler stand-in that still gives a real AdaBoost
comparison point, this uses the same mean+std-aggregated MFCC/Mel/chroma
vectors as the XGBoost baseline (via baseline_utils.load_flat_features).

Uses sample_weight to handle class imbalance, computed the same way
train.py computes it for the neural models (sklearn's "balanced"
scheme). AdaBoostClassifier has no class_weight parameter of its own
(unlike most sklearn classifiers) -- sample_weight passed to .fit() is
the way to get the same effect. Without this, an earlier run of this
script on the real (~9:1 imbalanced) ASVspoof data collapsed to
predicting "fake" for every sample: that still scores ~90% accuracy
purely from the class imbalance, while actually having learned nothing
(specificity was exactly 0.0). Accuracy alone doesn't catch this --
it's exactly the kind of failure Table 4's other metrics (specificity,
EER) exist to surface.
"""

import csv
import json

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config
from metrics_utils import compute_all_metrics, print_metrics
from baseline_utils import load_flat_features


def save_predictions_csv(y_true, y_pred, y_score_fake, model_name):
    """See evaluate.py's save_predictions_csv for the full explanation --
    same format, needed so statistical_analysis.py can compare this
    baseline against the hybrid model on the same test samples."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"predictions_{model_name}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_class", "y_score_fake"])
        for t, p, s in zip(y_true, y_pred, y_score_fake):
            writer.writerow([int(t), int(p), float(s)])
    print(f"Saved per-sample predictions to {out_path}")


def main():
    X_train, y_train = load_flat_features("train")
    X_test, y_test = load_flat_features("test")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    y_score_fake = model.predict_proba(X_test)[:, 1]

    save_predictions_csv(y_test, y_pred, y_score_fake, "adaboost")

    results = compute_all_metrics(y_test, y_pred, y_score_fake)
    print_metrics(results, title="AdaBoost baseline -- Evaluation results:")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "adaboost_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
