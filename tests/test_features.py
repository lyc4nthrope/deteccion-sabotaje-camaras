"""Tests for the classic-module descriptors (T30, T31, T32, T33, T37)."""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import features

ROOT = Path(__file__).resolve().parents[1]
BREMEN_ROOT = ROOT / "data" / "raw" / "bremen"
SPLIT = ROOT / "data" / "splits" / "split.csv"


def _texture(seed: int, size=(360, 640)) -> np.ndarray:
    """Random rectangles and circles: plenty of ORB keypoints."""
    rng = np.random.default_rng(seed)
    image = np.full((*size, 3), 128, np.uint8)
    for _ in range(150):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + int(rng.integers(5, 60)), y + int(rng.integers(5, 60))), color, -1)
        else:
            cv2.circle(image, (x, y), int(rng.integers(3, 30)), color, -1)
    return image


def _uniform(bgr=(0, 0, 255), size=(360, 640)) -> np.ndarray:
    return np.full((*size, 3), bgr, np.uint8)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- preprocessing (T30)


def test_preprocess_resizes_to_the_working_size_and_returns_gray():
    pre = features.preprocess(_texture(0, size=(720, 1280)))
    assert pre["bgr"].shape == (features.HEIGHT, features.WIDTH, 3)
    assert pre["gray"].shape == (features.HEIGHT, features.WIDTH) and pre["gray"].dtype == np.uint8
    assert pre["smooth"].shape == pre["gray"].shape


def test_preprocess_smooth_is_a_gaussian_blur_of_gray():
    pre = features.preprocess(_texture(0))
    assert np.abs(pre["smooth"].astype(int) - pre["gray"].astype(int)).mean() > 0
    assert cv2.Laplacian(pre["smooth"], cv2.CV_64F).var() < cv2.Laplacian(pre["gray"], cv2.CV_64F).var()


def test_load_image_fails_clearly_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        features.load_image(tmp_path / "missing.jpg")


# ---------------------------------------------------------------- sharpness (T31)


def test_sharpness_drops_when_the_image_is_blurred():
    gray = cv2.cvtColor(_texture(1), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 6)
    assert features.sharpness(gray) > 10 * features.sharpness(blurred)


def test_sharpness_of_a_uniform_image_is_zero():
    assert features.sharpness(np.full((50, 50), 90, np.uint8)) == 0.0


# ---------------------------------------------------------------- uniformity (T32)


def test_uniformity_of_a_flat_image_is_zero():
    std, entropy = features.uniformity(np.full((50, 50), 200, np.uint8))
    assert std == 0.0 and entropy == 0.0


def test_uniformity_of_random_noise_is_high():
    noise = np.random.default_rng(0).integers(0, 256, (300, 300), dtype=np.uint8)
    std, entropy = features.uniformity(noise)
    assert std > 70 and entropy > 7.9


def test_uniformity_std_has_the_exact_population_value():
    # Three pixels at 0 and one at 100: mean 25, std = sqrt((3 * 25**2 + 75**2) / 4) = 43.30.
    std, _ = features.uniformity(np.array([[0, 0, 0, 100]], np.uint8))
    assert std == pytest.approx(43.30127, rel=1e-5)


def test_entropy_of_two_equal_halves_is_one_bit():
    image = np.zeros((10, 10), np.uint8)
    image[:, 5:] = 255
    assert features.uniformity(image)[1] == pytest.approx(1.0)


# ---------------------------------------------------------------- color (T37)


def test_color_of_a_pure_red_cover_is_saturated_and_has_a_single_hue():
    color = features.color_descriptors(_uniform((0, 0, 255)))
    assert color["sat_mean"] == pytest.approx(1.0)
    assert color["hue_std"] == pytest.approx(0.0, abs=1e-6)


def test_color_of_a_gray_image_has_no_saturation_and_no_hue_dispersion():
    color = features.color_descriptors(_uniform((120, 120, 120)))
    assert color["sat_mean"] == 0.0 and color["hue_std"] == 0.0 and color["skin_ratio"] == 0.0


def test_hue_dispersion_grows_with_different_colors():
    image = _uniform((0, 0, 255))
    image[:, 320:] = (0, 255, 0)
    assert features.color_descriptors(image)["hue_std"] > 40


def test_hue_dispersion_is_circular_around_red():
    # Hues 2 and 178 (OpenCV scale 0-179) are both red, 4 units apart on the circle.
    hsv = np.zeros((10, 20, 3), np.uint8)
    hsv[..., 1:] = 255
    hsv[:, :10, 0], hsv[:, 10:, 0] = 2, 178
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert features.color_descriptors(image)["hue_std"] < 6


def test_skin_ratio_detects_a_hand_colored_patch():
    skin = cv2.cvtColor(np.full((50, 50, 3), (12, 120, 200), np.uint8), cv2.COLOR_HSV2BGR)
    blue = _uniform((255, 0, 0), size=(50, 50))
    assert features.color_descriptors(skin)["skin_ratio"] == pytest.approx(1.0)
    assert features.color_descriptors(blue)["skin_ratio"] == 0.0


# ---------------------------------------------------------------- geometry (T33)


@pytest.fixture(scope="module")
def matcher():
    return features.ReferenceMatcher([features.preprocess(_texture(10))["gray"]])


def test_orb_uses_the_class_configuration():
    assert features.ORB_FEATURES == 1000 and features.LOWE_RATIO == 0.75


def test_same_scene_has_many_matches_and_no_rotation(matcher):
    result = matcher.match(features.preprocess(_texture(10))["gray"])
    assert result["orb_matches"] > 200
    assert abs(result["angle"]) < 2


def test_rotated_scene_reports_the_rotation_angle(matcher):
    result = matcher.match(features.preprocess(_rotate(_texture(10), 30))["gray"])
    assert result["orb_matches"] > 50
    assert abs(abs(result["angle"]) - 30) < 3


def test_different_scene_has_few_matches(matcher):
    same = matcher.match(features.preprocess(_texture(10))["gray"])["orb_matches"]
    other = matcher.match(features.preprocess(_texture(99))["gray"])["orb_matches"]
    assert other < same / 5


def test_uniform_image_has_no_matches_and_no_angle(matcher):
    result = matcher.match(features.preprocess(_uniform())["gray"])
    assert result["orb_matches"] == 0 and math.isnan(result["angle"])


def test_matcher_keeps_the_best_reference():
    refs = [features.preprocess(_texture(s))["gray"] for s in (20, 21)]
    matcher = features.ReferenceMatcher(refs)
    result = matcher.match(features.preprocess(_texture(21))["gray"])
    assert result["best_reference"] == 1


def test_matcher_needs_at_least_one_reference():
    with pytest.raises(ValueError):
        features.ReferenceMatcher([])


# ---------------------------------------------------------------- full vector


def test_extract_returns_every_descriptor(matcher):
    vector = features.extract(_texture(10), matcher)
    assert set(vector) == {
        "lap_var", "gray_std", "entropy", "sat_mean", "hue_std", "skin_ratio",
        "orb_matches", "orb_inliers", "angle", "best_reference",
    }


# ---------------------------------------------------------------- real dataset


@pytest.mark.skipif(not (BREMEN_ROOT.exists() and SPLIT.exists()), reason="Bremen dataset not available")
def test_descriptors_separate_the_real_classes():
    split = pd.read_csv(SPLIT)
    refs = pd.read_csv(ROOT / "data" / "splits" / "referencias.csv")
    matcher = features.ReferenceMatcher(
        [features.preprocess(features.load_image(BREMEN_ROOT / p))["gray"] for p in refs["path"]]
    )
    train = split[split["set"] == "train"]
    medians = {}
    for label, group in train.groupby("label"):
        sample = group.sample(12, random_state=0)
        rows = [features.extract(features.load_image(BREMEN_ROOT / p), matcher) for p in sample["path"]]
        medians[label] = pd.DataFrame(rows).median()
    # Defocus removes sharp edges.
    assert medians["desenfocada"]["lap_var"] < medians["normal"]["lap_var"] / 3
    # A covered lens shows a single color. Intensity std does NOT separate it from normal night images
    # (dark and low contrast); the circular hue dispersion does (see docs/descriptores.md).
    assert medians["tapada"]["hue_std"] < medians["normal"]["hue_std"] / 3
    # A camera pointed elsewhere no longer matches the normal reference views.
    assert medians["girada"]["orb_matches"] < medians["normal"]["orb_matches"] / 2
    assert medians["normal"]["orb_matches"] > 50
