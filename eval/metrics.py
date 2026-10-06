"""Metric helpers. Thin wrappers over sklearn so the harness stays readable."""
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, cohen_kappa_score,
                             classification_report, confusion_matrix)


def classification_scores(y_true, y_pred, labels=None):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted",
                                      labels=labels, zero_division=0), 4),
    }


def per_class_report(y_true, y_pred, labels=None):
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)


def escalation_scores(y_true, y_pred):
    """Treat 'escalate' as the positive class."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "escalate_precision": round(precision_score(
            y_true, y_pred, pos_label="escalate", zero_division=0), 4),
        "escalate_recall": round(recall_score(
            y_true, y_pred, pos_label="escalate", zero_division=0), 4),
        "escalate_f1": round(f1_score(
            y_true, y_pred, pos_label="escalate", zero_division=0), 4),
    }


def agreement(human, judge):
    """Cohen's kappa + exact agreement between two raters on the same items."""
    exact = sum(1 for a, b in zip(human, judge) if a == b) / max(1, len(human))
    return {
        "exact_agreement": round(exact, 4),
        "cohen_kappa": round(cohen_kappa_score(human, judge), 4),
        "n": len(human),
    }


def numeric_agreement(human, judge):
    """Agreement for ordinal 1-5 ratings: quadratic-weighted kappa (ordinal-aware),
    plus a binarized 'acceptable' (>=4) kappa and mean absolute error."""
    import numpy as np
    h, j = np.array(human), np.array(judge)
    hb = ["good" if x >= 4 else "bad" for x in h]
    jb = ["good" if x >= 4 else "bad" for x in j]
    try:
        qwk = round(cohen_kappa_score(h, j, weights="quadratic"), 4)
    except Exception:  # noqa: BLE001  degenerate (all one value)
        qwk = None
    return {
        "quadratic_weighted_kappa": qwk,
        "acceptable_kappa": round(cohen_kappa_score(hb, jb), 4),
        "mean_abs_error": round(float(np.mean(np.abs(h - j))), 4),
        "n": len(human),
    }
