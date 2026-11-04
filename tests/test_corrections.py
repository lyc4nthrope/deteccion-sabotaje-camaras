"""Tests for the correction filters (T91)."""

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


def test_registry_has_the_demo_corrections():
    assert set(C.CORRECTIONS) == {"clahe", "gamma_inverso_2", "mediana", "gaussiano", "bilateral"}
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
