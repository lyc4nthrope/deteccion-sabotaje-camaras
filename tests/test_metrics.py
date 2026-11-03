"""Tests for the shared evaluation metrics (T25). Expected values are computed by hand."""

import math
import time

import numpy as np
import pytest

from src import metrics

Y_TRUE = ["normal", "normal", "normal", "normal", "desenfocada", "desenfocada", "girada", "tapada"]
Y_PRED = ["normal", "normal", "girada", "tapada", "desenfocada", "normal", "girada", "tapada"]


def test_labels_order_is_fixed():
    assert metrics.LABELS == ["normal", "desenfocada", "girada", "tapada"]


def test_confusion_matrix_rows_are_true_and_columns_are_predicted():
    cm = metrics.confusion(Y_TRUE, Y_PRED)
    assert list(cm.index) == metrics.LABELS and list(cm.columns) == metrics.LABELS
    assert cm.loc["normal", "normal"] == 2
    assert cm.loc["normal", "girada"] == 1 and cm.loc["normal", "tapada"] == 1
    assert cm.loc["desenfocada", "normal"] == 1
    assert cm.values.sum() == len(Y_TRUE)


def test_confusion_matrix_keeps_classes_without_samples():
    cm = metrics.confusion(["normal", "normal"], ["normal", "normal"])
    assert cm.shape == (4, 4) and cm.loc["girada"].sum() == 0


def test_f1_macro_averages_classes_equally():
    # Per class F1: normal P=2/3 R=2/4 -> 4/7; desenfocada P=1 R=1/2 -> 2/3; girada P=1/2 R=1 -> 2/3;
    # tapada P=1/2 R=1 -> 2/3. Macro = (4/7 + 3 * 2/3) / 4.
    assert metrics.f1_macro(Y_TRUE, Y_PRED) == pytest.approx((4 / 7 + 2) / 4)


def test_f1_macro_counts_classes_missing_in_both_as_zero_free():
    # Only classes present in y_true or y_pred enter the average.
    assert metrics.f1_macro(["normal", "girada"], ["normal", "girada"]) == pytest.approx(1.0)


def test_false_positive_rate_is_share_of_normals_flagged_as_sabotage():
    # 4 true normals, 2 predicted as some sabotage class.
    assert metrics.false_positive_rate(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_false_positive_rate_ignores_errors_between_sabotage_classes():
    assert metrics.false_positive_rate(["normal", "girada"], ["normal", "tapada"]) == 0.0


def test_false_positive_rate_needs_normal_samples():
    with pytest.raises(ValueError):
        metrics.false_positive_rate(["girada"], ["normal"])


def test_to_binary_marks_any_sabotage_as_one():
    assert metrics.to_binary(["normal", "desenfocada", "girada", "tapada"]).tolist() == [0, 1, 1, 1]


def test_binary_roc_perfect_and_inverted_scores():
    y = ["normal", "normal", "girada", "tapada"]
    assert metrics.binary_roc(y, [0.1, 0.2, 0.8, 0.9])["auc"] == pytest.approx(1.0)
    assert metrics.binary_roc(y, [0.9, 0.8, 0.2, 0.1])["auc"] == pytest.approx(0.0)


def test_binary_roc_known_value_and_curve_endpoints():
    # Sabotage scores 0.4, 0.8; normal scores 0.3, 0.6 -> 3 of 4 pairs ordered correctly: AUC = 0.75.
    roc = metrics.binary_roc(["normal", "normal", "girada", "girada"], [0.3, 0.6, 0.4, 0.8])
    assert roc["auc"] == pytest.approx(0.75)
    assert roc["fpr"][0] == 0.0 and roc["tpr"][-1] == 1.0 and roc["fpr"][-1] == 1.0


def test_binary_roc_needs_both_classes():
    with pytest.raises(ValueError):
        metrics.binary_roc(["girada", "tapada"], [0.1, 0.2])


def test_time_per_image_measures_milliseconds():
    def slow(_image):
        time.sleep(0.01)

    result = metrics.time_per_image(slow, [None, None, None])
    assert result["n"] == 3
    assert 9.0 <= result["mean_ms"] < 60.0
    assert result["std_ms"] >= 0.0


def test_time_per_image_needs_images():
    with pytest.raises(ValueError):
        metrics.time_per_image(lambda image: image, [])


def test_summary_collects_the_rubric_metrics():
    summary = metrics.summary(Y_TRUE, Y_PRED, scores=np.linspace(0, 1, len(Y_TRUE)))
    assert set(summary) == {"accuracy", "f1_macro", "fpr_normal", "auc_binaria"}
    assert summary["accuracy"] == pytest.approx(5 / 8)
    assert summary["fpr_normal"] == pytest.approx(0.5)


def test_summary_without_scores_has_no_auc():
    assert math.isnan(metrics.summary(Y_TRUE, Y_PRED)["auc_binaria"])
