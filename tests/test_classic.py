"""Tests for the classic rule-based classifier: rules, calibration and sabotage score (T40, T41, T42)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import classic, metrics

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTORS = ROOT / "results" / "clasico" / "descriptores_ubicacion1.csv"


def _rows(label, n, **values):
    return pd.DataFrame([{"label": label, **values}] * n)


@pytest.fixture
def train():
    """Separable synthetic descriptors that mimic the real class medians."""
    base = {"gray_std": 30.0, "entropy": 6.8, "sat_mean": 0.2, "skin_ratio": 0.05, "orb_inliers": 0, "angle": 0.1}
    return pd.concat(
        [
            _rows("normal", 40, **{**base, "orb_matches": 200, "lap_var": 400.0, "hue_std": 40.0}),
            _rows("normal", 10, **{**base, "orb_matches": 90, "lap_var": 40.0, "hue_std": 25.0}),
            # Dark night normals: low sharpness but the scene still matches. They must not move the blur
            # threshold, which is learnt only among sabotaged images.
            _rows("normal", 80, **{**base, "orb_matches": 150, "lap_var": 4.5, "hue_std": 20.0}),
            _rows("desenfocada", 20, **{**base, "orb_matches": 1, "lap_var": 5.0, "hue_std": 18.0, "gray_std": 16.0}),
            _rows("girada", 10, **{**base, "orb_matches": 8, "lap_var": 600.0, "hue_std": 70.0}),
            # Same sharpness as the defocused ones (they overlap in the real data): only color separates them.
            _rows("tapada", 20, **{**base, "orb_matches": 0, "lap_var": 5.0, "hue_std": 1.5, "gray_std": 8.0}),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------- stump


def test_best_threshold_separates_two_groups():
    values = np.array([1, 2, 3, 10, 11, 12], float)
    positive = np.array([0, 0, 0, 1, 1, 1], bool)
    threshold, direction, score = classic.best_threshold(values, positive)
    assert threshold == pytest.approx(6.5) and direction == ">=" and score == pytest.approx(1.0)


def test_best_threshold_finds_the_lower_side():
    values = np.array([1, 2, 3, 10, 11, 12], float)
    positive = np.array([1, 1, 1, 0, 0, 0], bool)
    threshold, direction, score = classic.best_threshold(values, positive)
    assert 3 < threshold <= 10 and direction == "<" and score == pytest.approx(1.0)


def test_best_threshold_uses_balanced_accuracy():
    # 100 negatives spread over 0..99 and 5 positives in the middle (50..58). Plain accuracy would put the
    # threshold at the top and call everything negative (99 of 105 right); balanced accuracy keeps the
    # 5 positives: threshold just below 50 with direction >=.
    values = np.r_[np.arange(100.0), [50.5, 52.5, 54.5, 56.5, 58.5]]
    positive = np.r_[np.zeros(100, bool), np.ones(5, bool)]
    threshold, direction, score = classic.best_threshold(values, positive)
    assert 49 < threshold <= 50.5 and direction == ">=" and score == pytest.approx(0.75, abs=0.01)


# ---------------------------------------------------------------- calibration


def test_calibration_learns_each_rule_from_train(train):
    thresholds = classic.calibrate(train)
    assert 8 < thresholds.match_min <= 90
    assert 5 < thresholds.blur_max <= 600
    assert thresholds.cover_feature in classic.COVER_CANDIDATES


def test_blur_threshold_is_learnt_only_from_sabotaged_images(train):
    # Midpoint between the sharpest blurry/covered sabotage (5) and the softest rotated one (600). The 80 dark
    # night normals (lap_var 4.5) must not pull the threshold down.
    assert classic.calibrate(train).blur_max == pytest.approx(302.5)


def test_calibrated_rules_classify_the_training_data(train):
    thresholds = classic.calibrate(train)
    assert (classic.predict(train, thresholds) == train["label"]).all()


def test_calibration_needs_every_class(train):
    with pytest.raises(ValueError):
        classic.calibrate(train[train["label"] != "tapada"])


# ---------------------------------------------------------------- rules


@pytest.fixture
def thresholds():
    return classic.Thresholds(
        match_min=50, blur_max=15.0, cover_feature="hue_std", cover_threshold=8.0, cover_direction="<", angle_max=1.0
    )


def _one(**values):
    base = {"orb_matches": 0, "lap_var": 0.0, "hue_std": 0.0, "gray_std": 0.0, "entropy": 0.0, "sat_mean": 0.0,
            "angle": 0.0}
    return pd.DataFrame([{**base, **values}])


def test_rule_enough_matches_is_normal_even_if_blurry(thresholds):
    assert classic.predict(_one(orb_matches=50, lap_var=1.0), thresholds)[0] == "normal"


def test_rule_few_matches_sharp_is_rotated(thresholds):
    assert classic.predict(_one(orb_matches=49, lap_var=15.0, hue_std=60), thresholds)[0] == "girada"


def test_rule_few_matches_blurry_single_color_is_covered(thresholds):
    assert classic.predict(_one(orb_matches=0, lap_var=3.0, hue_std=2.0), thresholds)[0] == "tapada"


def test_rule_few_matches_blurry_several_colors_is_defocused(thresholds):
    assert classic.predict(_one(orb_matches=0, lap_var=3.0, hue_std=8.0), thresholds)[0] == "desenfocada"


def test_rule_cover_direction_can_be_greater_or_equal():
    t = classic.Thresholds(
        match_min=50, blur_max=15.0, cover_feature="sat_mean", cover_threshold=0.5, cover_direction=">=", angle_max=1.0
    )
    assert classic.predict(_one(lap_var=3.0, sat_mean=0.9), t)[0] == "tapada"
    assert classic.predict(_one(lap_var=3.0, sat_mean=0.1), t)[0] == "desenfocada"


def test_rule_matching_scene_with_a_rotation_beyond_normal_is_rotated(thresholds):
    # Partial pan (Location 2): the scene still matches but the view turned.
    assert classic.predict(_one(orb_matches=80, lap_var=300.0, angle=-3.5), thresholds)[0] == "girada"


def test_rule_turned_view_is_rotated_even_if_it_looks_soft(thresholds):
    # The scene matches, so low sharpness is not a defocus sabotage: the decisive fact is the rotation.
    assert classic.predict(_one(orb_matches=80, lap_var=5.0, hue_std=2.0, angle=3.0), thresholds)[0] == "girada"


def test_rule_small_rotation_or_unknown_angle_stays_normal(thresholds):
    assert classic.predict(_one(orb_matches=80, lap_var=300.0, angle=0.5), thresholds)[0] == "normal"
    assert classic.predict(_one(orb_matches=80, lap_var=300.0, angle=float("nan")), thresholds)[0] == "normal"


def test_angle_threshold_is_the_99th_percentile_of_normal_train_angles(train):
    angles = np.linspace(0, 2.0, 101)  # |angle| 0.00, 0.02, ..., 2.00 -> 99th percentile = 1.98
    normal = pd.DataFrame({"label": "normal", "angle": np.r_[angles[:50], -angles[50:]]})
    others = train[train["label"] != "normal"].assign(angle=45.0)  # must not influence the threshold
    t = classic.calibrate(pd.concat([train.assign(angle=np.nan)[train["label"] == "normal"], normal, others]))
    assert t.angle_max == pytest.approx(1.98)


# ---------------------------------------------------------------- score and persistence


def test_score_is_half_at_the_threshold_and_decreases_with_matches(thresholds):
    scores = classic.sabotage_score(pd.DataFrame({"orb_matches": [0, 25, 50, 200], "angle": 0.0}), thresholds)
    assert scores[0] == pytest.approx(1.0) and scores[2] == pytest.approx(0.5)
    assert np.all(np.diff(scores) < 0) and scores.min() > 0


@pytest.mark.parametrize("values", [{"orb_matches": 10}, {"orb_matches": 80, "angle": 3.0}, {"orb_matches": 80, "angle": 0.2}])
def test_score_agrees_with_the_normal_decision(thresholds, values):
    frame = _one(lap_var=300.0, **values)
    assert (classic.sabotage_score(frame, thresholds)[0] > 0.5) == (classic.predict(frame, thresholds)[0] != "normal")


def test_score_grows_with_the_rotation_angle(thresholds):
    frame = pd.DataFrame({"orb_matches": [200, 200, 200], "angle": [0.0, 2.0, -10.0]})
    scores = classic.sabotage_score(frame, thresholds)
    assert scores[0] < scores[1] < scores[2]


def test_thresholds_round_trip(tmp_path, thresholds):
    path = tmp_path / "umbrales.json"
    classic.save_thresholds(thresholds, path)
    assert classic.load_thresholds(path) == thresholds


# ---------------------------------------------------------------- real descriptors


@pytest.mark.skipif(not DESCRIPTORS.exists(), reason="Run python -m src.build_features first")
def test_real_validation_performance_is_reasonable():
    descriptors = pd.read_csv(DESCRIPTORS)
    thresholds = classic.calibrate(descriptors[descriptors["set"] == "train"])
    val = descriptors[descriptors["set"] == "val"]
    predicted = classic.predict(val, thresholds)
    assert metrics.f1_macro(val["label"], predicted) > 0.8
    assert metrics.false_positive_rate(val["label"], predicted) < 0.15
