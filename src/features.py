"""Classic-module descriptors (T30, T31, T32, T33, T37).

Every image is reduced to a small vector of numbers that describe its sharpness, uniformity, color and how
well it matches the normal view of the camera. The rules of T40 classify from this vector.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

# Working size: half of the 1280x720 originals keeps the aspect ratio and speeds up every descriptor.
WIDTH, HEIGHT = 640, 360

# ORB configuration from class 11 (notebook "Algoritmos de extracción de características", ORB cell).
ORB_FEATURES = 1000
LOWE_RATIO = 0.75
MIN_MATCHES_FOR_GEOMETRY = 6
RANSAC_THRESHOLD = 5.0

# HSV thresholds (OpenCV scale: H 0-179, S and V 0-255).
COLOR_MIN_SAT = 40  # below this a pixel is gray and has no meaningful hue
COLOR_MIN_VAL = 40  # below this a pixel is too dark to have a meaningful hue
SKIN_HUE_MAX = 25  # skin tones: red-orange hues (0-50 degrees)
SKIN_SAT_RANGE = (40, 200)
SKIN_VAL_MIN = 60


def load_image(path: Path | str) -> np.ndarray:
    """Read a BGR image; fail clearly when the file cannot be read."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def preprocess(bgr: np.ndarray) -> dict:
    """Resize to the working size and derive the gray and HSV versions (T30).

    No smoothing here: a Gaussian blur lowered the margin of the sharpness rule on clean images and did not
    protect it against noise either (docs/clasico.md, section 7); noise correction belongs to T91.
    """
    resized = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return {
        "bgr": resized,
        "gray": gray,
        "hsv": cv2.cvtColor(resized, cv2.COLOR_BGR2HSV),
    }


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (T31): low when there are no sharp edges, as in a defocused camera."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def uniformity(gray: np.ndarray) -> tuple[float, float]:
    """Intensity standard deviation and histogram entropy in bits (T32): both near 0 for a covered lens."""
    histogram = np.bincount(gray.ravel(), minlength=256) / gray.size
    probabilities = histogram[histogram > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return float(gray.std()), abs(entropy)


def color_descriptors(bgr: np.ndarray) -> dict:
    """Color thresholding in HSV (T37); see color_from_hsv."""
    return color_from_hsv(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))


def color_from_hsv(hsv: np.ndarray) -> dict:
    """Color descriptors from an HSV image (T37).

    - sat_mean: mean saturation (0-1). A cover of one vivid color (the red cloth of Location 1) is high.
    - hue_std: circular standard deviation of the hue, in degrees, over colored pixels. It is ~0 for a single
      color and exactly 0 when there are no colored pixels at all (a gray or very dark image).
    - skin_ratio: share of pixels with skin tones, as when a hand covers the camera.
    """
    hue, sat, val = (hsv[..., i].astype(float) for i in range(3))

    colored = (sat >= COLOR_MIN_SAT) & (val >= COLOR_MIN_VAL)
    if colored.any():
        angles = np.deg2rad(hue[colored] * 2.0)
        resultant = min(1.0, float(np.abs(np.exp(1j * angles).mean())))
        hue_std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))) if resultant > 0 else 180.0
    else:
        hue_std = 0.0

    skin = (hue <= SKIN_HUE_MAX) & (sat >= SKIN_SAT_RANGE[0]) & (sat <= SKIN_SAT_RANGE[1]) & (val >= SKIN_VAL_MIN)
    return {"sat_mean": float(sat.mean() / 255.0), "hue_std": float(hue_std), "skin_ratio": float(skin.mean())}


def estimate_rotation(source: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Rotation angle (degrees) of a RANSAC similarity transform from source to target points.

    Returns (nan, inliers) when fewer than MIN_MATCHES_FOR_GEOMETRY points agree: an angle supported by two or
    three points is meaningless (it gave values like 150 degrees on normal images).
    """
    transform, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD
    )
    if transform is None:
        return float("nan"), 0
    count = int(inliers.sum())
    if count < MIN_MATCHES_FOR_GEOMETRY:
        return float("nan"), count
    return math.degrees(math.atan2(transform[1, 0], transform[0, 0])), count


class ReferenceMatcher:
    """ORB matching against the normal reference views of one camera (T33).

    For each test image it keeps the reference with the most good matches (Lowe ratio test) and, when there
    are enough matches, estimates the rotation between both views with a RANSAC similarity transform.
    """

    def __init__(self, references_gray: Sequence[np.ndarray]):
        if len(references_gray) == 0:
            raise ValueError("At least one reference image is needed.")
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.references = [self.orb.detectAndCompute(gray, None) for gray in references_gray]

    def _good_matches(self, descriptors: np.ndarray, reference_descriptors: np.ndarray | None) -> list:
        if reference_descriptors is None or len(reference_descriptors) < 2:
            return []
        pairs = self.matcher.knnMatch(descriptors, reference_descriptors, k=2)
        return [p[0] for p in pairs if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]

    def match(self, gray: np.ndarray) -> dict:
        result = {"orb_matches": 0, "orb_inliers": 0, "angle": float("nan"), "best_reference": -1}
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return result

        candidates = [self._good_matches(descriptors, ref_desc) for _, ref_desc in self.references]
        best = int(np.argmax([len(c) for c in candidates]))
        good = candidates[best]
        result["orb_matches"], result["best_reference"] = len(good), best
        if len(good) < MIN_MATCHES_FOR_GEOMETRY:
            return result

        reference_keypoints = self.references[best][0]
        source = np.float32([reference_keypoints[m.trainIdx].pt for m in good])
        target = np.float32([keypoints[m.queryIdx].pt for m in good])
        result["angle"], result["orb_inliers"] = estimate_rotation(source, target)
        return result


def extract(bgr: np.ndarray, matcher: ReferenceMatcher) -> dict:
    """Full descriptor vector of one image."""
    pre = preprocess(bgr)
    std, entropy = uniformity(pre["gray"])
    return {
        "lap_var": sharpness(pre["gray"]),
        "gray_std": std,
        "entropy": entropy,
        **color_from_hsv(pre["hsv"]),
        **matcher.match(pre["gray"]),
    }
