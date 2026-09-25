"""
statistical_analysis.py

Statistical significance testing for the deepfake voice detection thesis.
Compares the hybrid CNN+BiLSTM model against each baseline (CNN-only,
RNN-only, XGBoost, AdaBoost) using three complementary techniques:

1. Bootstrap confidence intervals
   -> "How much would this metric plausibly vary if we'd drawn a
      different test set from the same distribution?"

2. McNemar's test
   -> "Do model A and model B disagree in a way that's unlikely to be
      chance, given the *same* test samples?"

3. Cohen's g (matched-pairs effect size) + bootstrap CI on the raw
   metric difference
   -> "Setting aside statistical significance, how *large* is the
      difference between the two models, in a way that doesn't just
      grow with test-set size?" Cohen's g comes straight out of
      McNemar's own contingency table; the bootstrap CI on the raw
      difference (e.g. "+10.1 points of recall, 95% CI [9.6, 10.6]")
      gives the practical size in the metric's own units.

Why all three, not just one:
- p-values (McNemar) tell you whether a difference is likely real, but
  not how big it is.
- Confidence intervals tell you the plausible range of a single model's
  performance, but not whether two models differ significantly.
- Effect size (Cohen's d) tells you the magnitude of a difference,
  independent of sample size, which matters because ASVspoof's test
  partition is large enough that even tiny, practically meaningless
  differences can come out "significant".
Reporting all three is standard practice for defensible ML comparisons
(this combination follows the general approach recommended by
Dietterich (1998) and is widely used in bio-medical / speech ML papers).

-------------------------------------------------------------------------
EXPECTED INPUT FORMAT
-------------------------------------------------------------------------
One CSV per model, each with these columns, one row per test-set sample,
in a *consistent sample order* across all files (this is critical for
McNemar's test and paired bootstrap -- row i must refer to the same
audio sample in every file). This is exactly what evaluate.py,
xgboost_baseline.py, and adaboost_baseline.py now write via their
save_predictions_csv() helper:

    true_label,predicted_class,y_score_fake
    0,0,0.0002
    1,1,1.0
    ...

y_score_fake is P(class == 1 == "fake"), NOT the probability of
whichever class was predicted -- this matters because roc_auc_score
needs a score for a fixed positive class, not the "winning" class's own
confidence. This mirrors the convention metrics_utils.compute_all_metrics()
already uses project-wide.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python statistical_analysis.py

Edit the CONFIG section below to point at your actual results directory
and baseline names first. Output: a JSON report + a printable summary
table, written to RESULTS_DIR/statistical_analysis/.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

# Only needed for the exact/corrected McNemar implementation.
# pip install statsmodels --break-system-packages   (if not already installed)
from statsmodels.stats.contingency_tables import mcnemar

# Reuse the project's own EER implementation rather than reimplementing it
# a second time -- keeps this number identical in definition to Table 4's
# EER wherever this script runs alongside the rest of the pipeline (i.e.
# in the same directory as metrics_utils.py).
from metrics_utils import compute_eer


# =========================================================================
# CONFIG -- edit these to match your project (see config.py conventions)
# =========================================================================

RESULTS_DIR = "/content/drive/MyDrive/deepfake_detector/results"

# Filename template for each model's saved predictions. {model} is filled
# in from MODEL_NAMES below. Adjust to whatever evaluate.py / 
# train_baselines.py actually write.
PRED_FILE_TEMPLATE = os.path.join(RESULTS_DIR, "predictions_{model}.csv")

# The main hybrid model's key -- matches evaluate.py's MODEL_REGISTRY key
# ("hybrid"), which is also the {model} substitution used by its
# save_predictions_csv() -> predictions_hybrid.csv
MAIN_MODEL = "hybrid"

# All baselines to compare the main model against. cnn_only/rnn_only come
# from evaluate.py; xgboost/adaboost come from their own baseline scripts.
BASELINE_MODELS = ["cnn_only", "rnn_only", "xgboost", "adaboost"]

# Metrics to bootstrap. Must be keys in METRIC_FUNCS below. Matches
# Table 4's metric set (metrics_utils.compute_all_metrics) so the
# statistical analysis lines up with the same numbers already reported
# elsewhere in the thesis.
METRICS_TO_REPORT = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "eer"]

N_BOOTSTRAP = 10000        # iterations for CI and Cohen's d
CI_LEVEL = 0.95
RANDOM_SEED = 42

OUTPUT_DIR = os.path.join(RESULTS_DIR, "statistical_analysis")

# Metrics where a LOWER value is better (everything else: higher is better).
# EER is the one exception in this project's metric set -- used to set the
# "favors" direction in metric_difference_bootstrap below.
LOWER_IS_BETTER = {"eer"}


# =========================================================================
# Metric functions -- each takes (y_true, y_pred, y_prob) and returns a
# single float. y_prob may be None if not needed/available.
# =========================================================================

def _safe_auc(y_true, y_pred, y_prob):
    if y_prob is None:
        return np.nan
    # AUC (and EER, below) are undefined if a bootstrap resample happens
    # to contain only one class -- vanishingly unlikely given the real
    # class balance and n~thousands, but guarded rather than left to
    # crash a 10k-iteration loop.
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def _safe_eer(y_true, y_pred, y_prob):
    if y_prob is None:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan
    return compute_eer(y_true, y_prob)


def _specificity(y_true, y_pred, y_prob):
    # labels=[0,1] forces a full 2x2 matrix even if a bootstrap resample
    # happens to contain only one class -- without it, .ravel() would
    # raise on a 1x1 matrix instead of returning a clean 0.0/nan.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


METRIC_FUNCS = {
    "accuracy": lambda yt, yp, ypr: accuracy_score(yt, yp),
    "precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
    "recall": lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
    "specificity": _specificity,
    "f1": lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
    "auc": _safe_auc,
    "eer": _safe_eer,
}


# =========================================================================
# 1. Bootstrap confidence intervals
# =========================================================================

def bootstrap_ci(y_true, y_pred, y_prob=None, metric="accuracy",
                  n_iterations=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_SEED):
    """
    Resample the test set (with replacement) n_iterations times and
    recompute `metric` each time. Returns the point estimate on the full
    set plus the [lower, upper] percentile CI from the bootstrap
    distribution.

    Why resample the *test set indices* rather than assume a normal
    distribution: with only ~thousands of test samples and metrics like
    F1 that aren't simple means, the bootstrap gives a more honest
    (non-parametric) interval than a Wald/normal-approximation CI.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob) if y_prob is not None else None
    n = len(y_true)
    metric_fn = METRIC_FUNCS[metric]

    point_estimate = metric_fn(y_true, y_pred, y_prob)

    scores = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, n)  # sample n indices with replacement
        yt, yp = y_true[idx], y_pred[idx]
        ypr = y_prob[idx] if y_prob is not None else None
        scores[i] = metric_fn(yt, yp, ypr)

    alpha = 1 - ci
    lower = np.nanpercentile(scores, 100 * (alpha / 2))
    upper = np.nanpercentile(scores, 100 * (1 - alpha / 2))

    return {
        "metric": metric,
        "point_estimate": float(point_estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "n_iterations": n_iterations,
    }


# =========================================================================
# 2. McNemar's test
# =========================================================================

def mcnemars_test(y_true, pred_a, pred_b):
    """
    Tests whether two models -- evaluated on the *same* test samples --
    disagree asymmetrically more than chance would predict.

    Builds the 2x2 contingency table of (A correct, B correct):

                        B correct   B wrong
        A correct         n11         n10
        A wrong           n01         n00

    Only the discordant cells (n10, n01) drive the test: if A and B were
    equally good, their sets of "which samples do I get wrong" should be
    interchangeable, so n10 ~ n01. A large asymmetry -> low p-value ->
    the difference in error patterns is unlikely to be chance.

    Uses the exact binomial test when either discordant cell is small
    (<25, standard rule of thumb), otherwise the chi-square version with
    continuity correction.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    n11 = int(np.sum(a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    n01 = int(np.sum(~a_correct & b_correct))
    n00 = int(np.sum(~a_correct & ~b_correct))

    table = [[n11, n10], [n01, n00]]
    exact = min(n10, n01) < 25
    result = mcnemar(table, exact=exact, correction=not exact)

    return {
        "contingency_table": {"n11_both_correct": n11, "n10_only_A_correct": n10,
                               "n01_only_B_correct": n01, "n00_both_wrong": n00},
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "test_type": "exact_binomial" if exact else "chi_square_corrected",
        "significant_at_0.05": bool(result.pvalue < 0.05),
        "cohens_g": cohens_g(n10, n01),
    }


def cohens_g(n10, n01):
    """
    Effect size for a matched-pairs (McNemar) comparison -- the textbook
    pairing for McNemar's test (Cohen, 1988), computed directly from the
    same discordant-pair counts McNemar's test already uses. Unlike a
    bootstrap-based standardized mean difference, this does NOT grow with
    sample size, which is what an effect size is supposed to guarantee.

    g = P(A right, B wrong | exactly one is right) - 0.5

    Ranges from -0.5 (B wins every discordant pair) to +0.5 (A wins every
    discordant pair); 0 means the two models split discordant pairs evenly.
    Conventional thresholds (Cohen, 1988): |g| ~ 0.05 small, ~0.15 medium,
    ~0.25 large.
    """
    total_discordant = n10 + n01
    if total_discordant == 0:
        return 0.0
    p = n10 / total_discordant
    return p - 0.5


# =========================================================================
# 3. Bootstrap CI on the raw metric difference
# =========================================================================

def metric_difference_bootstrap(y_true, pred_a, prob_a, pred_b, prob_b,
                                 metric="accuracy", n_iterations=N_BOOTSTRAP,
                                 ci=CI_LEVEL, seed=RANDOM_SEED):
    """
    How much do model A and model B actually differ on `metric`, in the
    metric's own units (not a standardized effect size)? Uses the same
    paired-bootstrap logic as before -- same resampled indices applied to
    both models each iteration -- but reports the point estimate and CI
    of (score_a - score_b) directly, rather than dividing by the
    bootstrap distribution's tiny standard deviation (which is what
    produced the earlier, misleadingly huge "Cohen's d" values: with
    ~25k test samples the bootstrap std is inherently small, so that
    division inflated the number and made it scale with sample size --
    the opposite of what an effect size should do). This is the
    "practical size of the difference" number; cohens_g (see
    mcnemars_test) is the sample-size-independent effect size to report
    alongside it.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    prob_a = np.asarray(prob_a) if prob_a is not None else None
    prob_b = np.asarray(prob_b) if prob_b is not None else None
    n = len(y_true)
    metric_fn = METRIC_FUNCS[metric]

    point_a = metric_fn(y_true, pred_a, prob_a)
    point_b = metric_fn(y_true, pred_b, prob_b)
    point_diff = point_a - point_b

    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, n)  # same indices for both models
        yt = y_true[idx]
        score_a = metric_fn(yt, pred_a[idx], prob_a[idx] if prob_a is not None else None)
        score_b = metric_fn(yt, pred_b[idx], prob_b[idx] if prob_b is not None else None)
        diffs[i] = score_a - score_b

    alpha = 1 - ci
    lower = np.nanpercentile(diffs, 100 * (alpha / 2))
    upper = np.nanpercentile(diffs, 100 * (1 - alpha / 2))

    if metric in LOWER_IS_BETTER:
        favors = "tie" if point_diff == 0 else ("B" if point_diff > 0 else "A")
    else:
        favors = "tie" if point_diff == 0 else ("A" if point_diff > 0 else "B")

    return {
        "metric": metric,
        "point_diff_A_minus_B": float(point_diff),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "favors": favors,
    }


# =========================================================================
# Orchestration
# =========================================================================

@dataclass
class ModelPredictions:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def load_predictions(model_name):
    path = PRED_FILE_TEMPLATE.format(model=model_name)
    df = pd.read_csv(path)
    return ModelPredictions(
        name=model_name,
        y_true=df["true_label"].to_numpy(),
        y_pred=df["predicted_class"].to_numpy(),
        y_prob=df["y_score_fake"].to_numpy() if "y_score_fake" in df.columns else None,
    )


def compare_pair(main, other):
    """Run all three tests comparing `main` against `other`."""
    assert len(main.y_true) == len(other.y_true), (
        f"Sample count mismatch between {main.name} ({len(main.y_true)}) and "
        f"{other.name} ({len(other.y_true)}) -- predictions must be over the "
        f"same test set in the same row order."
    )
    assert np.array_equal(main.y_true, other.y_true), (
        f"true_label columns differ between {main.name} and {other.name} -- "
        f"check that both were evaluated on identical, identically-ordered "
        f"test samples."
    )

    report = {"model_a": main.name, "model_b": other.name}

    report["bootstrap_ci"] = {
        main.name: {m: bootstrap_ci(main.y_true, main.y_pred, main.y_prob, metric=m)
                    for m in METRICS_TO_REPORT},
        other.name: {m: bootstrap_ci(other.y_true, other.y_pred, other.y_prob, metric=m)
                     for m in METRICS_TO_REPORT},
    }

    report["mcnemar"] = mcnemars_test(main.y_true, main.y_pred, other.y_pred)

    report["metric_differences"] = {
        m: metric_difference_bootstrap(main.y_true, main.y_pred, main.y_prob,
                                        other.y_pred, other.y_prob, metric=m)
        for m in METRICS_TO_REPORT
    }

    return report


def print_summary(report):
    a, b = report["model_a"], report["model_b"]
    print(f"\n{'=' * 70}\n{a}  vs  {b}\n{'=' * 70}")

    print("\n-- Bootstrap 95% CIs --")
    for model_name in (a, b):
        print(f"  {model_name}:")
        for m, res in report["bootstrap_ci"][model_name].items():
            print(f"    {m:10s}: {res['point_estimate']:.4f}  "
                  f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")

    mc = report["mcnemar"]
    print("\n-- McNemar's test (error-pattern disagreement) --")
    print(f"  contingency table: {mc['contingency_table']}")
    print(f"  statistic={mc['statistic']:.4f}  p={mc['p_value']:.4g}  "
          f"({mc['test_type']})  significant={mc['significant_at_0.05']}")
    print(f"  Cohen's g (matched-pairs effect size): {mc['cohens_g']:+.4f}  "
          f"(|g| ~0.05 small, ~0.15 medium, ~0.25 large)")

    print("\n-- Metric differences (A - B), with bootstrap 95% CI --")
    for m, res in report["metric_differences"].items():
        print(f"  {m:10s}: {res['point_diff_A_minus_B']:+.4f}  "
              f"[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]  favors {res['favors']}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main_preds = load_predictions(MAIN_MODEL)
    all_reports = []

    for baseline_name in BASELINE_MODELS:
        baseline_preds = load_predictions(baseline_name)
        report = compare_pair(main_preds, baseline_preds)
        print_summary(report)
        all_reports.append(report)

    out_path = os.path.join(OUTPUT_DIR, "statistical_analysis_report.json")
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
