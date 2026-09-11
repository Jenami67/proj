"""XGBoost baseline (Section 3.1.6 comparative baselines; Bird & Lotfi
2023 used a similar temporal-feature + gradient-boosting approach).

XGBoost can't take a variable-length time sequence like the CNN/RNN model
does, so each clip's MFCC + Mel + chroma sequences are collapsed into one
fixed-length vector (mean + std of each feature dimension over time) via
baseline_utils.load_flat_features -- the standard, simple way to feed
frame-level audio features into a classical ML model.
"""

import json

from xgboost import XGBClassifier

import config
from metrics_utils import compute_all_metrics, print_metrics
from baseline_utils import load_flat_features


def main():
    X_train, y_train = load_flat_features("train")
    X_test, y_test = load_flat_features("test")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score_fake = model.predict_proba(X_test)[:, 1]

    results = compute_all_metrics(y_test, y_pred, y_score_fake)
    print_metrics(results, title="XGBoost baseline -- Evaluation results:")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "xgboost_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
