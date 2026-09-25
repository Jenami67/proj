"""XGBoost baseline (Section 3.1.6 comparative baselines; Bird & Lotfi
2023 used a similar temporal-feature + gradient-boosting approach).

XGBoost can't take a variable-length time sequence like the CNN/RNN model
does, so each clip's MFCC + Mel + chroma sequences are collapsed into one
fixed-length vector (mean + std of each feature dimension over time) via
baseline_utils.load_flat_features -- the standard, simple way to feed
frame-level audio features into a classical ML model.

Uses scale_pos_weight to handle class imbalance -- XGBoost's own native
mechanism for binary classification, computed here as
count(negative_class) / count(positive_class) per the library's
documented convention. Given this project's label convention (0 = real/
bonafide, 1 = fake/spoof) and real being the minority class in ASVspoof
LA (~9:1 fake:real), this ratio comes out below 1.0, correctly telling
XGBoost to weight the minority (real) class more heavily -- the same
effect train.py's class_weight="balanced" has for the neural models.
Without this, an earlier run on the real data showed a specificity of
only 0.38 (62% of genuine real speech misclassified as fake) despite a
superficially fine-looking 92.8% accuracy -- again a case where
accuracy alone masks a real imbalance problem.
"""

import csv
import json

import numpy as np
from xgboost import XGBClassifier

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

    n_negative = int(np.sum(y_train == 0))  # real / bonafide
    n_positive = int(np.sum(y_train == 1))  # fake / spoof
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.4f} ({n_negative} real / {n_positive} fake)")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score_fake = model.predict_proba(X_test)[:, 1]

    save_predictions_csv(y_test, y_pred, y_score_fake, "xgboost")

    results = compute_all_metrics(y_test, y_pred, y_score_fake)
    print_metrics(results, title="XGBoost baseline -- Evaluation results:")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "xgboost_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
