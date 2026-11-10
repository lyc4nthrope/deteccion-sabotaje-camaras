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


@pytest.mark.parametrize("bad", [["Normal", "girada"], ["normal", 3]])
def test_metrics_reject_unknown_labels(bad):
    with pytest.raises(ValueError, match="Unknown"):
        metrics.confusion(bad, ["normal", "girada"])
    with pytest.raises(ValueError, match="Unknown"):
        metrics.f1_macro(["normal", "girada"], bad)


def test_time_per_image_discards_a_warm_up_call():
    calls = []

    def predict(_image):
        if not calls:
            time.sleep(0.2)  # slow first call: lazy initialisation
        calls.append(1)

    result = metrics.time_per_image(predict, [None, None])
    assert len(calls) == 3 and result["n"] == 2
    assert result["mean_ms"] < 50 and "median_ms" in result


# ---------------------------------------------------------------- T26: per-class metrics (class 20)


def test_binary_metrics_reproduce_the_class_20_example():
    # Slide 28 of class 20: TP = 40, TN = 50, FP = 10, FN = 20.
    m = metrics.binary_metrics(tp=40, tn=50, fp=10, fn=20)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.80)
    assert m["recall"] == pytest.approx(40 / 60)
    assert m["specificity"] == pytest.approx(50 / 60)
    assert m["fpr"] == pytest.approx(10 / 60)
    assert m["f1"] == pytest.approx(2 * 0.8 * (2 / 3) / (0.8 + 2 / 3))


def test_binary_metrics_give_zero_instead_of_dividing_by_zero():
    m = metrics.binary_metrics(tp=0, tn=5, fp=0, fn=0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["specificity"] == 1.0 and m["fpr"] == 0.0


def test_per_class_report_matches_hand_computed_values():
    # Y_TRUE / Y_PRED defined at the top of this file. One-vs-rest counts per class:
    # normal: TP 2, FP 1, FN 2, TN 3 -> precision 2/3, recall 1/2, specificity 3/4
    # desenfocada: TP 1, FP 0, FN 1, TN 6 -> precision 1, recall 1/2, specificity 1
    # girada: TP 1, FP 1, FN 0, TN 6 -> precision 1/2, recall 1, specificity 6/7
    # tapada: TP 1, FP 1, FN 0, TN 6 -> precision 1/2, recall 1, specificity 6/7
    report = metrics.per_class_report(Y_TRUE, Y_PRED)
    assert report.loc["normal", "precision"] == pytest.approx(2 / 3)
    assert report.loc["normal", "recall"] == pytest.approx(1 / 2)
    assert report.loc["normal", "specificity"] == pytest.approx(3 / 4)
    assert report.loc["desenfocada", "specificity"] == pytest.approx(1.0)
    assert report.loc["girada", "specificity"] == pytest.approx(6 / 7)
    assert report.loc["tapada", "f1"] == pytest.approx(2 / 3)
    assert report.loc["normal", "support"] == 4 and report.loc["girada", "support"] == 1


def test_macro_average_is_the_plain_mean_of_the_classes():
    report = metrics.per_class_report(Y_TRUE, Y_PRED)
    classes = report.loc[metrics.LABELS]
    for column in ("precision", "recall", "specificity", "f1"):
        assert report.loc["macro", column] == pytest.approx(classes[column].mean())
    assert report.loc["macro", "f1"] == pytest.approx(metrics.f1_macro(Y_TRUE, Y_PRED))


def test_micro_average_pools_the_counts():
    # Single-label multiclass: pooled precision = pooled recall = accuracy (5/8).
    # Pooled specificity = sum TN / sum (TN + FP) = (3 + 6 + 6 + 6) / (21 + 3) = 21/24.
    report = metrics.per_class_report(Y_TRUE, Y_PRED)
    assert report.loc["micro", "precision"] == pytest.approx(5 / 8)
    assert report.loc["micro", "recall"] == pytest.approx(5 / 8)
    assert report.loc["micro", "f1"] == pytest.approx(5 / 8)
    assert report.loc["micro", "specificity"] == pytest.approx(21 / 24)


def test_per_class_report_shows_why_accuracy_misleads_on_imbalanced_data():
    # Class 20, slide 29: 95 % negatives and a model that always says "normal".
    y_true = ["normal"] * 95 + ["girada"] * 5
    y_pred = ["normal"] * 100
    report = metrics.per_class_report(y_true, y_pred)
    assert metrics.summary(y_true, y_pred)["accuracy"] == pytest.approx(0.95)
    assert report.loc["girada", "recall"] == 0.0
    assert report.loc["macro", "f1"] < 0.5


def test_per_class_report_keeps_only_present_classes_and_rejects_unknown_labels():
    report = metrics.per_class_report(["normal", "girada"], ["normal", "girada"])
    assert list(report.index) == ["normal", "girada", "macro", "micro"]
    with pytest.raises(ValueError, match="Unknown"):
        metrics.per_class_report(["Normal"], ["normal"])
