"""Shared evaluation metrics (Table 4), used by evaluate.py and both
baseline scripts so the numbers are computed identically everywhere."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)


def compute_eer(y_true, y_scores):
    """Equal Error Rate: the point where the false acceptance rate equals
    the false rejection rate. Not built into scikit-learn -- computed here
    from the ROC curve, following the standard definition used in
    anti-spoofing research (Todisco et al., 2019)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[eer_index] + fnr[eer_index]) / 2


def compute_all_metrics(y_true, y_pred, y_score_fake):
    """y_score_fake: predicted probability of the 'fake' class for each sample."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score_fake),
        "eer": compute_eer(y_true, y_score_fake),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(results, title="Evaluation results"):
    print(f"\n{title}")
    for key, value in results.items():
        print(f"  {key}: {value}")
