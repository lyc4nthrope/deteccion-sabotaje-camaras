"""Correction filters that undo light and noise degradation (T91).

Each function takes a uint8 image (gray or BGR) and returns a uint8 image of the same shape. They are the
defense against the perturbations of T20: CLAHE for darkness and low contrast, inverse gamma for gamma shifts,
median for salt-and-pepper noise, and Gaussian or bilateral smoothing for Gaussian noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def _check_uint8(image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization.

    For color images it works on the L channel of LAB, so hues are not changed. It restores contrast in dark
    scenes (brightness perturbation, T20).
    """
    _check_uint8(image)
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    if image.ndim == 2:
        return clahe_op.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe_op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law correction out = 255 * (in / 255) ** gamma.

    To undo a darkening perturbation with gamma g (T20), use gamma = 1 / g. Values below 1 brighten.
    """
    _check_uint8(image)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    normalized = image.astype(np.float32) / 255.0
    return np.clip(np.round(255.0 * np.power(normalized, gamma)), 0, 255).astype(np.uint8)


def median(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter. The nonlinear answer to salt-and-pepper noise (T20)."""
    _check_uint8(image)
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be an odd number >= 3")
    return cv2.medianBlur(image, ksize)


def gaussian(image: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """Gaussian smoothing, the linear answer to Gaussian noise (T20)."""
    _check_uint8(image)
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be an odd number >= 3")
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def bilateral(image: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Edge-preserving smoothing: it averages neighbours that are close in both space and intensity."""
    _check_uint8(image)
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


# Corrections the demo can toggle (T85): name -> function(image).
CORRECTIONS = {
    "clahe": clahe,
    "gamma_inverso_2": lambda image: gamma_correct(image, 1 / 2.0),
    "mediana": median,
    "gaussiano": gaussian,
    "bilateral": bilateral,
}
