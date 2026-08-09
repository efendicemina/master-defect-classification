"""Fixed-label metric calculation for Phase A1."""

from __future__ import annotations

from typing import Any


def classification_metrics(
    y_true: list[str], y_pred: list[str], labels: tuple[str, ...]
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }
