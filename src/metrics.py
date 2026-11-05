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


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Metrics of a 2x2 confusion matrix, with the formulas of class 20 (slides 25-27).

    A division by zero gives 0 (for example, the precision of a model that never predicts positive).
    """
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": _ratio(tn, tn + fp),
        "fpr": _ratio(fp, fp + tn),
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def per_class_report(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Precision, recall, specificity, F1 and support per class (one vs rest), plus macro and micro rows (T26).

    - macro: plain mean over the classes, so a rare class weighs as much as a frequent one;
    - micro: the counts of every class are pooled before computing each metric. In single-label multiclass
      classification micro precision, recall and F1 all equal the accuracy.
    Only classes present in y_true or y_pred are reported, as in f1_macro.
    """
    _check_labels(y_true, y_pred)
    present = [label for label in LABELS if label in set(y_true) | set(y_pred)]
    matrix = confusion(y_true, y_pred).loc[present, present].to_numpy()
    total = matrix.sum()
    rows, pooled = {}, np.zeros(4)
    for i, label in enumerate(present):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        tn = total - tp - fp - fn
        pooled += (tp, tn, fp, fn)
        m = binary_metrics(tp, tn, fp, fn)
        rows[label] = {k: m[k] for k in ("precision", "recall", "specificity", "f1")} | {"support": int(tp + fn)}
    report = pd.DataFrame.from_dict(rows, orient="index")
    columns = ["precision", "recall", "specificity", "f1"]
    macro = report[columns].mean().to_dict() | {"support": int(total)}
    micro = {k: v for k, v in binary_metrics(*pooled).items() if k in columns} | {"support": int(total)}
    report.loc["macro"] = macro
    report.loc["micro"] = micro
    report["support"] = report["support"].astype(int)
    return report[columns + ["support"]]


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
