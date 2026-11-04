"""Shared evaluation helpers, used by both train.py and app.py.

Keeping the metric logic in ONE place guarantees the numbers in the README
(from train.py) and the numbers in the Streamlit app are computed identically.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "default_next_month"          # canonical target name used everywhere
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def get_scores(model, X):
    """Continuous score for the positive class (needed for AUC / ROC).

    Uses predict_proba when the model has it (LR, Tree, KNN, NB) and falls back
    to decision_function (SVM without probability=True).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X, y):
    """Return (metrics_dict, y_pred, scores). Never calls .fit()."""
    y_pred = model.predict(X)
    scores = get_scores(model, X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return {k: float(v) for k, v in metrics.items()}, y_pred, np.asarray(scores)
