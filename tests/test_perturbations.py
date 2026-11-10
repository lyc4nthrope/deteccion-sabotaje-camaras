"""Tests for the light and noise perturbations (T20, T21)."""

import numpy as np
import pytest

from src import perturbations as P


def _scene(seed: int = 0, shape=(60, 80, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _flat(value: int, shape=(60, 80, 3)) -> np.ndarray:
    return np.full(shape, value, np.uint8)


ALL_CASES = [(name, level) for name in P.PERTURBATIONS for level in P.LEVELS]


# ---------------------------------------------------------------- contract (T21)


@pytest.mark.parametrize("name, level", ALL_CASES)
def test_perturbation_keeps_shape_and_uint8_dtype(name, level):
    image = _scene(1)
    out = P.apply(name, image, level, seed=3)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name, level", ALL_CASES)
def test_perturbation_keeps_grayscale_images(name, level):
    image = _scene(2, shape=(60, 80))
    out = P.apply(name, image, level, seed=3)
    assert out.shape == image.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name, level", ALL_CASES)
def test_perturbation_is_reproducible_with_the_same_seed(name, level):
    image = _scene(4)
    first = P.apply(name, image, level, seed=7)
    second = P.apply(name, image, level, seed=7)
    assert np.array_equal(first, second)


def test_random_perturbations_change_with_a_different_seed():
    image = _scene(5)
    for name in ("ruido_gaussiano", "sal_pimienta"):
        assert not np.array_equal(P.apply(name, image, 2, seed=1), P.apply(name, image, 2, seed=2))


def test_input_image_is_not_modified():
    image = _scene(6)
    original = image.copy()
    for name, level in ALL_CASES:
        P.apply(name, image, level, seed=0)
    assert np.array_equal(image, original)


def test_there_is_no_blur_perturbation():
    # D7: blur is the "desenfocada" class, not a degradation of normal images.
    assert not any("blur" in name or "desenfoq" in name for name in P.PERTURBATIONS)


@pytest.mark.parametrize("bad_level", [0, 4, -1])
def test_invalid_level_is_rejected(bad_level):
    with pytest.raises(ValueError):
        P.brightness(_scene(), bad_level)


def test_non_uint8_input_is_rejected():
    with pytest.raises(ValueError):
        P.gamma(np.zeros((10, 10, 3), np.float32), 1)


def test_unknown_perturbation_name_is_rejected():
    with pytest.raises(ValueError):
        P.apply("desenfoque", _scene(), 1)


# ---------------------------------------------------------------- brightness and gamma


def test_darken_scales_the_intensity_by_the_level_factor():
    out = P.brightness(_flat(200), 2, mode="darken")
    assert int(out[0, 0, 0]) == round(200 * P.BRIGHTNESS_FACTORS["darken"][1])


def test_lighten_saturates_at_255():
    out = P.brightness(_flat(200), 3, mode="lighten")
    assert int(out.max()) == 255


def test_darken_gets_stronger_with_the_level():
    image = _scene(7)
    means = [P.brightness(image, level, "darken").mean() for level in P.LEVELS]
    assert means[0] > means[1] > means[2]


def test_brightness_rejects_unknown_mode():
    with pytest.raises(ValueError):
        P.brightness(_scene(), 1, mode="blur")


def test_gamma_darkens_midtones_with_the_power_law():
    # 255 * (128 / 255) ** 2 = 64.25, which rounds to 64 for level 2 (gamma 2.0).
    out = P.gamma(_flat(128), 2)
    assert int(out[0, 0, 0]) == 64


def test_gamma_keeps_black_and_white_fixed():
    out = P.gamma(np.array([[0, 255]], np.uint8), 3)
    assert out.tolist() == [[0, 255]]


def test_gamma_gets_stronger_with_the_level():
    image = _scene(8)
    means = [P.gamma(image, level).mean() for level in P.LEVELS]
    assert means[0] > means[1] > means[2]


# ---------------------------------------------------------------- noise


def test_gaussian_noise_std_grows_with_the_level():
    image = _flat(128)
    stds = [(P.gaussian_noise(image, level, seed=0).astype(float) - 128).std() for level in P.LEVELS]
    assert stds[0] < stds[1] < stds[2]


def test_gaussian_noise_std_is_close_to_the_configured_sigma():
    image = _flat(128, shape=(200, 200))
    noisy = P.gaussian_noise(image, 2, seed=0).astype(float)
    assert noisy.std() == pytest.approx(P.GAUSSIAN_SIGMA[1], rel=0.05)


def test_salt_and_pepper_changes_about_the_configured_share_of_pixels():
    image = _flat(128, shape=(200, 200))
    out = P.salt_and_pepper(image, 2, seed=0)
    changed = (out[..., 0] != 128).mean()
    assert changed == pytest.approx(P.SALT_PEPPER_AMOUNT[1], abs=0.01)


def test_salt_and_pepper_only_produces_black_or_white_pixels():
    out = P.salt_and_pepper(_flat(128), 3, seed=0)
    assert set(np.unique(out)) <= {0, 128, 255}


def test_salt_and_pepper_splits_the_noise_between_salt_and_pepper():
    out = P.salt_and_pepper(_flat(128, shape=(200, 200)), 3, seed=0)
    salt = (out == 255).mean()
    pepper = (out == 0).mean()
    assert salt == pytest.approx(pepper, abs=0.01)


# ---------------------------------------------------------------- JPEG


def test_jpeg_error_grows_with_the_level():
    image = _scene(9, shape=(120, 160, 3))
    errors = [np.abs(P.jpeg(image, level).astype(float) - image).mean() for level in P.LEVELS]
    assert errors[0] < errors[1] < errors[2]


def test_jpeg_of_a_flat_image_stays_flat():
    out = P.jpeg(_flat(100), 3)
    assert np.abs(out.astype(int) - 100).max() <= 2
