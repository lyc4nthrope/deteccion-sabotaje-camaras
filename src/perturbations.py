"""Perturbations that simulate light and noise degradation for the robustness audit (T20).

Five perturbation types, three severity levels each (D7). Blur is deliberately absent: it is a sabotage class
("desenfocada"), so a defocused input is not a degraded normal image. Every function takes a uint8 image and
returns a uint8 image of the same shape. Random perturbations take a seed, so the same seed gives the same output.
"""

from __future__ import annotations

import cv2
import numpy as np

LEVELS = (1, 2, 3)

# Brightness: multiplicative gain. Darkening simulates dusk or a dim room, lightening simulates backlight.
BRIGHTNESS_FACTORS = {"darken": (0.7, 0.5, 0.3), "lighten": (1.3, 1.6, 2.0)}

# Gamma > 1 darkens the midtones (underexposed sensor); the three values are the darkening levels.
GAMMA_VALUES = (1.5, 2.0, 3.0)

# Gaussian noise standard deviation, in gray levels (0-255).
GAUSSIAN_SIGMA = (5.0, 15.0, 30.0)

# Share of pixels replaced by salt (255) or pepper (0); half of the amount goes to each.
SALT_PEPPER_AMOUNT = (0.01, 0.05, 0.10)

# JPEG quality: lower means stronger compression artifacts.
JPEG_QUALITY = (60, 30, 10)


def _pick(values: tuple, level: int):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return values[level - 1]


def _check_uint8(image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def brightness(image: np.ndarray, level: int, mode: str = "darken") -> np.ndarray:
    """Scale the intensity by a factor (T20). mode is "darken" or "lighten"."""
    _check_uint8(image)
    if mode not in BRIGHTNESS_FACTORS:
        raise ValueError(f"mode must be one of {tuple(BRIGHTNESS_FACTORS)}, got {mode!r}")
    factor = _pick(BRIGHTNESS_FACTORS[mode], level)
    return _to_uint8(image.astype(np.float32) * factor)


def gamma(image: np.ndarray, level: int) -> np.ndarray:
    """Darken with the power law out = 255 * (in / 255) ** g, with g > 1 (T20)."""
    _check_uint8(image)
    exponent = _pick(GAMMA_VALUES, level)
    normalized = image.astype(np.float32) / 255.0
    return _to_uint8(255.0 * np.power(normalized, exponent))


def gaussian_noise(image: np.ndarray, level: int, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise with a fixed seed (T20)."""
    _check_uint8(image)
    sigma = _pick(GAUSSIAN_SIGMA, level)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape)
    return _to_uint8(image.astype(np.float32) + noise)


def salt_and_pepper(image: np.ndarray, level: int, seed: int = 0) -> np.ndarray:
    """Replace a share of pixels with 255 (salt) or 0 (pepper), with a fixed seed (T20)."""
    _check_uint8(image)
    amount = _pick(SALT_PEPPER_AMOUNT, level)
    rng = np.random.default_rng(seed)
    draw = rng.random(image.shape[:2])
    out = image.copy()
    out[draw < amount / 2] = 255
    out[(draw >= amount / 2) & (draw < amount)] = 0
    return out


def jpeg(image: np.ndarray, level: int) -> np.ndarray:
    """Encode to JPEG and decode back, which adds compression artifacts (T20)."""
    _check_uint8(image)
    quality = _pick(JPEG_QUALITY, level)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


# Registry used by the audit (T70): name -> function(image, level, seed).
PERTURBATIONS = {
    "brillo_oscuro": lambda image, level, seed=0: brightness(image, level, "darken"),
    "brillo_claro": lambda image, level, seed=0: brightness(image, level, "lighten"),
    "gamma": lambda image, level, seed=0: gamma(image, level),
    "ruido_gaussiano": gaussian_noise,
    "sal_pimienta": salt_and_pepper,
    "jpeg": lambda image, level, seed=0: jpeg(image, level),
}


def apply(name: str, image: np.ndarray, level: int, seed: int = 0) -> np.ndarray:
    """Apply a registered perturbation by name."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; expected one of {tuple(PERTURBATIONS)}")
    return PERTURBATIONS[name](image, level, seed)
