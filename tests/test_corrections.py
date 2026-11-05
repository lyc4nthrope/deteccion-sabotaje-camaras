"""Tests for the correction filters (T91)."""

import cv2
import numpy as np
import pytest

from src import corrections as C
from src import perturbations as P


def _scene(seed=0, shape=(60, 80, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _smooth(shape=(120, 160, 3)):
    # Smooth gradient: noise-free reference for checking filters.
    ramp = np.linspace(40, 200, shape[1], dtype=np.float32)
    plane = np.tile(ramp, (shape[0], 1))
    return np.stack([plane] * shape[2], axis=-1).astype(np.uint8) if len(shape) == 3 else plane.astype(np.uint8)


ALL_FILTERS = [
    ("clahe", C.clahe),
    ("gamma", lambda img: C.gamma_correct(img, 0.5)),
    ("mediana", C.median),
    ("gaussiano", C.gaussian),
    ("bilateral", C.bilateral),
]


# ---------------------------------------------------------------- contract (T91)


@pytest.mark.parametrize("name, fn", ALL_FILTERS)
def test_filter_keeps_shape_and_uint8_for_color(name, fn):
    image = _scene(1)
    out = fn(image)
    assert out.shape == image.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name, fn", ALL_FILTERS)
def test_filter_keeps_shape_and_uint8_for_gray(name, fn):
    image = _scene(2, shape=(60, 80))
    out = fn(image)
    assert out.shape == image.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name, fn", ALL_FILTERS)
def test_filter_rejects_non_uint8(name, fn):
    with pytest.raises(ValueError):
        fn(np.zeros((20, 20, 3), np.float32))


def test_registry_has_one_inverse_gamma_per_darkening_level():
    assert set(C.CORRECTIONS) == {
        "clahe", "gamma_inverso_nivel_1", "gamma_inverso_nivel_2", "gamma_inverso_nivel_3",
        "brillo_claro_inverso_nivel_1", "brillo_claro_inverso_nivel_2", "brillo_claro_inverso_nivel_3",
        "mediana", "gaussiano", "bilateral",
    }
    assert set(C.GAMMA_INVERSE) == {f"gamma_inverso_nivel_{k}" for k in P.LEVELS}


@pytest.mark.parametrize("level", P.LEVELS)
def test_each_inverse_gamma_undoes_its_own_level(level):
    original = _smooth()
    restored = C.CORRECTIONS[f"gamma_inverso_nivel_{level}"](P.gamma(original, level))
    assert np.abs(restored.astype(int) - original.astype(int)).mean() < 2
    image = _scene(3)
    for fn in C.CORRECTIONS.values():
        assert fn(image).shape == image.shape


def test_even_kernel_is_rejected():
    with pytest.raises(ValueError):
        C.median(_scene(), ksize=4)
    with pytest.raises(ValueError):
        C.gaussian(_scene(), ksize=2)


def test_gamma_must_be_positive():
    with pytest.raises(ValueError):
        C.gamma_correct(_scene(), 0)


# ---------------------------------------------------------------- effect on the perturbations


def test_clahe_lifts_the_brightness_of_a_dark_scene_without_shrinking_its_range():
    # Measured on the smooth ramp: mean 35.9 -> 41.0, range 48 -> 54 (clip limit 2.0 keeps the lift modest).
    dark = P.brightness(_smooth(), 3, mode="darken")
    restored = C.clahe(dark)
    assert restored.mean() > dark.mean() + 3
    assert int(restored.max()) - int(restored.min()) >= int(dark.max()) - int(dark.min())


def test_inverse_gamma_undoes_the_gamma_perturbation():
    # Perturbation level 2 uses gamma 2.0; the correction uses 1 / 2.0.
    original = _smooth()
    darkened = P.gamma(original, 2)
    restored = C.gamma_correct(darkened, 1 / 2.0)
    assert np.abs(restored.astype(int) - original.astype(int)).mean() < 2


def test_median_removes_salt_and_pepper_noise():
    original = _smooth()
    noisy = P.salt_and_pepper(original, 2, seed=0)
    fixed = C.median(noisy)
    error_before = np.abs(noisy.astype(int) - original.astype(int)).mean()
    error_after = np.abs(fixed.astype(int) - original.astype(int)).mean()
    assert error_after < error_before / 3


def test_gaussian_reduces_gaussian_noise():
    original = _smooth()
    noisy = P.gaussian_noise(original, 2, seed=0)
    smoothed = C.gaussian(noisy, ksize=5)
    error_before = np.abs(noisy.astype(int) - original.astype(int)).mean()
    error_after = np.abs(smoothed.astype(int) - original.astype(int)).mean()
    assert error_after < error_before


def test_bilateral_keeps_a_sharp_edge():
    image = np.zeros((40, 40), np.uint8)
    image[:, 20:] = 200
    out = C.bilateral(image)
    assert int(out[20, 19]) < 40 and int(out[20, 20]) > 160


# ---------------------------------------------------------------- overexposure (T92)


@pytest.mark.parametrize("level", P.LEVELS)
def test_unbrighten_restores_unsaturated_pixels(level):
    # Values 20..110: after the largest gain (2.0) they stay below 255, so nothing is clipped.
    original = np.linspace(20, 110, 120 * 160, dtype=np.float32).reshape(120, 160).astype(np.uint8)
    brightened = P.brightness(original, level, mode="lighten")
    restored = C.unbrighten(brightened, level)
    assert np.abs(restored.astype(int) - original.astype(int)).max() <= 2


def test_unbrighten_cannot_recover_clipped_pixels():
    # Documented limit: a pixel clipped at 255 by the perturbation has lost its value.
    original = np.full((20, 20), 200, np.uint8)
    brightened = P.brightness(original, 3, mode="lighten")
    assert int(brightened[0, 0]) == 255
    restored = C.unbrighten(brightened, 3)
    assert int(restored[0, 0]) < 200


def test_unbrighten_rejects_unknown_level():
    with pytest.raises(ValueError):
        C.unbrighten(_scene(), 4)


# ---------------------------------------------------------------- CLAHE keeps hue (T91, T37)


def _hue_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean circular hue difference (OpenCV scale 0-179) over pixels that have a clear colour."""
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    ha, sa = hsv_a[..., 0].astype(float), hsv_a[..., 1]
    hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 0].astype(float)
    colored = sa > 60
    diff = np.abs(ha - hb)
    diff = np.minimum(diff, 180 - diff)
    return float(diff[colored].mean())


def test_clahe_keeps_the_hue_of_a_color_image():
    # Low-contrast blue and red patches: CLAHE should change brightness only, not the hue.
    rng = np.random.default_rng(0)
    image = np.zeros((120, 160, 3), np.uint8)
    image[:, :80] = (150, 60, 40) + rng.integers(-6, 7, (120, 80, 3))
    image[:, 80:] = (40, 60, 150) + rng.integers(-6, 7, (120, 80, 3))
    image = np.clip(image, 0, 255).astype(np.uint8)
    # L-only CLAHE gives ~0.5; applying CLAHE per BGR channel gives ~2.0 on the same image, so 1.0 separates them.
    assert _hue_difference(image, C.clahe(image)) < 1.0
