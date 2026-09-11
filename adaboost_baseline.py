"""AdaBoost baseline (Section 3.1.6 comparative baselines; Nair et al.
2024 used AdaBoost with pause-based features).

We don't implement the pause-based features here (jitter/shimmer/pause
timing require extra libraries beyond Librosa -- flagged as a separate,
optional extension in the project notes, not needed for a first working
version). As a simpler stand-in that still gives a real AdaBoost
comparison point, this uses the same mean+std-aggregated MFCC/Mel/chroma
vectors as the XGBoost baseline (via baseline_utils.load_flat_features).
"""

import json

from sklearn.ensemble import AdaBoostClassifier

import config
from metrics_utils import compute_all_metrics, print_metrics
from baseline_utils import load_flat_features


def main():
    X_train, y_train = load_flat_features("train")
    X_test, y_test = load_flat_features("test")

    model = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score_fake = model.predict_proba(X_test)[:, 1]

    results = compute_all_metrics(y_test, y_pred, y_score_fake)
    print_metrics(results, title="AdaBoost baseline -- Evaluation results:")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "adaboost_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
