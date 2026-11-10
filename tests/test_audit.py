"""Tests for the robustness audit: degraded test sets and per-condition metrics (T70, T71)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import audit, classic, features

ROOT = Path(__file__).resolve().parents[1]


def _image(seed=0, size=(720, 1280)):
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


# ---------------------------------------------------------------- conditions (T70)


def test_conditions_are_the_original_plus_6_types_by_3_levels():
    conditions = audit.conditions()
    assert conditions[0] == ("original", 0)
    assert len(conditions) == 1 + 6 * 3
    assert {name for name, _ in conditions[1:]} == {
        "brillo_oscuro", "brillo_claro", "gamma", "ruido_gaussiano", "sal_pimienta", "jpeg",
    }
    assert {level for _, level in conditions[1:]} == {1, 2, 3}


def test_degrade_resizes_before_perturbing():
    degraded = audit.degrade(_image(), "ruido_gaussiano", 3, "a.jpg")
    assert degraded.shape == (features.HEIGHT, features.WIDTH, 3)
    resized = audit.resize(_image())
    # Noise added after the resize keeps its full strength (sigma 30 at level 3).
    assert np.std(degraded.astype(float) - resized.astype(float)) > 20


def test_original_condition_only_resizes():
    image = _image()
    assert np.array_equal(audit.degrade(image, "original", 0, "a.jpg"), audit.resize(image))


def test_degrade_is_reproducible_per_image_and_condition():
    image = _image()
    a = audit.degrade(image, "sal_pimienta", 2, "img_1.jpg")
    b = audit.degrade(image, "sal_pimienta", 2, "img_1.jpg")
    c = audit.degrade(image, "sal_pimienta", 2, "img_2.jpg")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_seed_is_stable_across_runs():
    # A fixed hash (not Python's salted hash()) so the CNN can regenerate the same images later.
    assert audit.seed_for("Location1_Normal/x.jpg", "jpeg", 1) == audit.seed_for("Location1_Normal/x.jpg", "jpeg", 1)
    assert audit.seed_for("Location1_Normal/x.jpg", "jpeg", 1) != audit.seed_for("Location1_Normal/x.jpg", "jpeg", 2)


def test_degraded_descriptors_cover_every_image_and_condition(monkeypatch, tmp_path):
    images = pd.DataFrame({"path": ["a.jpg", "b.jpg"], "label": ["normal", "girada"]})
    monkeypatch.setattr(audit.features, "load_image", lambda path: _image(seed=len(str(path))))
    matcher = features.ReferenceMatcher([audit.resize(_image(seed=99))[..., 0]])
    table = audit.degraded_descriptors(images, matcher, root=tmp_path)
    assert len(table) == 2 * 19
    assert set(zip(table["perturbacion"], table["nivel"])) == set(audit.conditions())
    assert {"lap_var", "orb_matches", "angle", "label", "path"} <= set(table.columns)


# ---------------------------------------------------------------- per-condition metrics (T71)


@pytest.fixture
def thresholds():
    return classic.Thresholds(
        match_min=50, blur_max=15.0, cover_feature="hue_std", cover_threshold=8.0, cover_direction="<", angle_max=1.0
    )


def _rows(perturbation, level, matches):
    base = {"lap_var": 300.0, "gray_std": 30.0, "entropy": 6.0, "sat_mean": 0.2, "hue_std": 40.0, "angle": 0.1}
    return [
        {**base, "perturbacion": perturbation, "nivel": level, "label": "normal", "orb_matches": m} for m in matches
    ] + [{**base, "perturbacion": perturbation, "nivel": level, "label": "girada", "orb_matches": 5}]


def test_condition_table_reports_one_row_per_condition_with_rubric_metrics(thresholds):
    frame = pd.DataFrame(_rows("original", 0, [200, 200]) + _rows("brillo_claro", 3, [200, 10]))
    table = audit.condition_table(frame, thresholds, model="clasico")
    assert list(table["perturbacion"]) == ["original", "brillo_claro"]
    assert {"modelo", "nivel", "n", "accuracy", "f1_macro", "fpr_normal", "auc_binaria"} <= set(table.columns)
    original, bright = table.iloc[0], table.iloc[1]
    assert original["fpr_normal"] == 0.0 and bright["fpr_normal"] == pytest.approx(0.5)
    assert original["n"] == 3 and (table["modelo"] == "clasico").all()


def test_condition_table_counts_predictions_per_class(thresholds):
    frame = pd.DataFrame(_rows("brillo_claro", 3, [200, 10]))
    row = audit.condition_table(frame, thresholds, model="clasico").iloc[0]
    assert row["pred_normal"] == 1 and row["pred_girada"] == 2


def test_condition_table_keeps_the_conditions_order(thresholds):
    frame = pd.DataFrame(_rows("jpeg", 1, [200]) + _rows("original", 0, [200]) + _rows("gamma", 2, [200]))
    table = audit.condition_table(frame, thresholds, model="clasico")
    assert list(zip(table["perturbacion"], table["nivel"])) == [("original", 0), ("gamma", 2), ("jpeg", 1)]
