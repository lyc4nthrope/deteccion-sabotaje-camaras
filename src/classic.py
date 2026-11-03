"""Classic rule-based classifier: rules, threshold calibration and sabotage score (T40, T41, T42).

Decision tree (same idea as the reference paper, thresholds learnt from the training set):

    orb_matches >= match_min                -> normal   (the scene still matches the normal references)
    otherwise, lap_var < blur_max           -> tapada if the cover rule holds, else desenfocada
    otherwise                               -> girada   (sharp image of a different scene)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import LABELS

# Descriptors tried to separate a covered lens from a defocused one; the best one on train is kept.
COVER_CANDIDATES = ("hue_std", "gray_std", "entropy", "sat_mean", "lap_var")


@dataclass(frozen=True)
class Thresholds:
    match_min: float
    blur_max: float
    cover_feature: str
    cover_threshold: float
    cover_direction: str  # "<" or ">=": side of cover_threshold that means "tapada"


def best_threshold(values: np.ndarray, positive: np.ndarray, directions=(">=", "<")) -> tuple[float, str, float]:
    """Threshold and direction that best separate positive from negative samples.

    Candidates are the midpoints between consecutive distinct values. The criterion is the balanced accuracy
    (mean of the hit rate of each group), so a small group weighs as much as a large one.
    """
    values, positive = np.asarray(values, float), np.asarray(positive, bool)
    distinct = np.unique(values)
    candidates = (distinct[:-1] + distinct[1:]) / 2 if len(distinct) > 1 else distinct
    best = (float(candidates[0]), directions[0], -1.0)
    for direction in directions:
        for threshold in candidates:
            predicted = values >= threshold if direction == ">=" else values < threshold
            score = (np.mean(predicted[positive]) + np.mean(~predicted[~positive])) / 2
            if score > best[2]:
                best = (float(threshold), direction, float(score))
    return best


def calibrate(train: pd.DataFrame) -> Thresholds:
    """Learn every threshold from the training descriptors (T41)."""
    missing = set(LABELS) - set(train["label"])
    if missing:
        raise ValueError(f"Calibration needs every class; missing: {sorted(missing)}")

    match_min, _, _ = best_threshold(train["orb_matches"], train["label"] == "normal", directions=(">=",))

    sabotage = train[train["label"] != "normal"]
    blur_max, _, _ = best_threshold(
        sabotage["lap_var"], sabotage["label"].isin(["desenfocada", "tapada"]), directions=("<",)
    )

    blurry = sabotage[sabotage["label"].isin(["desenfocada", "tapada"])]
    cover = max(
        ((feature, *best_threshold(blurry[feature], blurry["label"] == "tapada")) for feature in COVER_CANDIDATES),
        key=lambda option: option[3],
    )
    return Thresholds(match_min, blur_max, cover[0], cover[1], cover[2])


def predict(descriptors: pd.DataFrame, t: Thresholds) -> np.ndarray:
    """Apply the decision tree to every row (T40)."""
    normal = descriptors["orb_matches"].to_numpy() >= t.match_min
    blurry = descriptors["lap_var"].to_numpy() < t.blur_max
    feature = descriptors[t.cover_feature].to_numpy()
    covered = feature < t.cover_threshold if t.cover_direction == "<" else feature >= t.cover_threshold
    return np.where(normal, "normal", np.where(blurry, np.where(covered, "tapada", "desenfocada"), "girada"))


def sabotage_score(descriptors: pd.DataFrame, t: Thresholds) -> np.ndarray:
    """Continuous sabotage score in (0, 1] for the ROC curve (T42).

    match_min / (match_min + orb_matches): 1 with no matches, 0.5 exactly at the normal threshold and lower
    the better the image matches the normal references.
    """
    matches = descriptors["orb_matches"].to_numpy(dtype=float)
    return t.match_min / (t.match_min + matches)


def save_thresholds(t: Thresholds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(t), indent=2))


def load_thresholds(path: Path) -> Thresholds:
    return Thresholds(**json.loads(path.read_text()))
