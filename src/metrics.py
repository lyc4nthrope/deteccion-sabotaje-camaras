"""Shared evaluation metrics for the classic module, the CNN and the robustness audit (T25).

Every model is evaluated with these functions so the comparison between approaches is fair.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve

LABELS = ["normal", "desenfocada", "girada", "tapada"]
NORMAL = "normal"


def _check_labels(*groups: Iterable) -> None:
    unknown = {value for group in groups for value in group} - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(map(str, unknown))}. Valid labels: {LABELS}")


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> pd.DataFrame:
    """Confusion matrix with true classes as rows and predicted classes as columns, in a fixed order."""
    _check_labels(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


def f1_macro(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Unweighted mean of the per-class F1, over the classes present in y_true or y_pred.

    A rare class (girada) weighs the same as the frequent one (normal).
    """
    _check_labels(y_true, y_pred)
    present = [label for label in LABELS if label in set(y_true) | set(y_pred)]
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def false_positive_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Share of normal images predicted as any sabotage class (false alarms in a stable environment)."""
    _check_labels(y_true, y_pred)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    normal = y_true == NORMAL
    if not normal.any():
        raise ValueError("The false positive rate needs at least one normal image.")
    return float(np.mean(y_pred[normal] != NORMAL))


def to_binary(labels: Iterable[str]) -> np.ndarray:
    """1 for any sabotage class, 0 for normal."""
    return (np.asarray(list(labels)) != NORMAL).astype(int)


def binary_roc(y_true: Sequence[str], scores: Sequence[float]) -> dict:
    """ROC curve and AUC of normal (0) vs sabotage (1), given a score where higher means more sabotage."""
    y = to_binary(y_true)
    if y.min() == y.max():
        raise ValueError("The ROC curve needs normal and sabotaged images.")
    fpr, tpr, thresholds = roc_curve(y, np.asarray(scores, dtype=float))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": float(auc(fpr, tpr))}


def time_per_image(predict: Callable, images: Sequence) -> dict:
    """Mean, median and standard deviation, in milliseconds, of predict(image) over the given images.

    One warm-up call on the first image is discarded, so lazy initialisation does not inflate the result.
    """
    if len(images) == 0:
        raise ValueError("At least one image is needed to measure time.")
    predict(images[0])
    times = []
    for image in images:
        start = time.perf_counter()
        predict(image)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "n": len(times),
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
    }


def summary(y_true: Sequence[str], y_pred: Sequence[str], scores: Sequence[float] | None = None) -> dict:
    """Rubric metrics in one row: accuracy, macro F1, false positive rate on normals and binary AUC."""
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    auc_value = float("nan") if scores is None else binary_roc(y_true, scores)["auc"]
    return {
        "accuracy": float(np.mean(y_true_arr == y_pred_arr)),
        "f1_macro": f1_macro(y_true, y_pred),
        "fpr_normal": false_positive_rate(y_true, y_pred),
        "auc_binaria": auc_value,
    }
