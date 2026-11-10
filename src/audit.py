"""Robustness audit: degraded test sets and per-condition metrics (T70, T71).

Run from the repository root after python -m src.evaluate_classic:

    python -m src.audit

Every test image is read and reduced to the descriptor size (640x360) once, and only then perturbed: perturbing
the 4608x2592 originals and reducing afterwards would average the noise away. Degraded images are not stored:
they are regenerated from a fixed seed per image, perturbation and level, so the CNN (T71) can rebuild exactly the
same set. Outputs in results/auditoria/: descriptores_test_degradado.csv and tabla.csv.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import classic, data, features, metrics
from src import perturbations as P

OUT_DIR = data.RESULTS_DIR / "auditoria"
TIME_SAMPLE = 20
ORIGINAL = ("original", 0)


def conditions() -> list[tuple[str, int]]:
    """The original condition followed by every perturbation at every level (6 x 3)."""
    return [ORIGINAL] + [(name, level) for name in P.PERTURBATIONS for level in P.LEVELS]


def resize(image: np.ndarray) -> np.ndarray:
    """Reduce to the descriptor working size, as features.preprocess does."""
    return cv2.resize(image, (features.WIDTH, features.HEIGHT), interpolation=cv2.INTER_AREA)


def seed_for(path: str, name: str, level: int) -> int:
    """Stable seed for one image and condition (CRC32, not Python's per-process salted hash)."""
    return zlib.crc32(f"{path}|{name}|{level}".encode())


def degrade(image: np.ndarray, name: str, level: int, path: str) -> np.ndarray:
    """Resize first, then apply the perturbation with the seed of this image and condition."""
    small = resize(image)
    if (name, level) == ORIGINAL:
        return small
    return P.apply(name, small, level, seed_for(path, name, level))


def degraded_descriptors(
    images: pd.DataFrame, matcher: features.ReferenceMatcher, root: Path = data.BREMEN_ROOT
) -> pd.DataFrame:
    """Descriptor vector of every image under every condition (T70)."""
    rows = []
    for path, label in zip(images["path"], images["label"]):
        image = features.load_image(root / path)
        for name, level in conditions():
            vector = features.extract(degrade(image, name, level, path), matcher)
            rows.append({"path": path, "label": label, "perturbacion": name, "nivel": level, **vector})
    return pd.DataFrame(rows)


def condition_table(frame: pd.DataFrame, thresholds: classic.Thresholds, model: str) -> pd.DataFrame:
    """Rubric metrics of the classic module for each condition, in the order of conditions() (T71)."""
    order = {condition: i for i, condition in enumerate(conditions())}
    rows = []
    for (name, level), group in frame.groupby(["perturbacion", "nivel"], sort=False):
        predicted = classic.predict(group, thresholds)
        scores = classic.sabotage_score(group, thresholds)
        counts = pd.Series(predicted).value_counts()
        rows.append(
            {
                "modelo": model,
                "perturbacion": name,
                "nivel": int(level),
                "n": len(group),
                **metrics.summary(group["label"], predicted, scores),
                **{f"pred_{label}": int(counts.get(label, 0)) for label in metrics.LABELS},
            }
        )
    table = pd.DataFrame(rows)
    table["_order"] = [order[(n, l)] for n, l in zip(table["perturbacion"], table["nivel"])]
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def time_per_condition(images: pd.DataFrame, matcher: features.ReferenceMatcher, thresholds) -> pd.DataFrame:
    """Mean time per frame of descriptors + rules on a sample of degraded images, per condition."""
    sample = images.sample(TIME_SAMPLE, random_state=0)
    originals = {p: features.load_image(data.BREMEN_ROOT / p) for p in sample["path"]}
    rows = []
    for name, level in conditions():
        frames = [degrade(originals[p], name, level, p) for p in sample["path"]]
        timing = metrics.time_per_image(
            lambda f: classic.predict(pd.DataFrame([features.extract(f, matcher)]), thresholds), frames
        )
        rows.append({"perturbacion": name, "nivel": level, "ms_por_cuadro": timing["mean_ms"]})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = data.load_split()
    test = split[split["set"] == "test"].reset_index(drop=True)
    references = pd.read_csv(data.SPLITS_DIR / "referencias.csv")["path"]
    matcher = features.ReferenceMatcher(
        [features.preprocess(features.load_image(data.BREMEN_ROOT / p))["gray"] for p in references]
    )
    thresholds = classic.load_thresholds(data.RESULTS_DIR / "clasico" / "umbrales.json")

    descriptors = degraded_descriptors(test, matcher)
    descriptors.to_csv(OUT_DIR / "descriptores_test_degradado.csv", index=False)
    table = condition_table(descriptors, thresholds, model="clasico")
    table = table.merge(time_per_condition(test, matcher, thresholds), on=["perturbacion", "nivel"])
    table.to_csv(OUT_DIR / "tabla.csv", index=False)
    print(table.drop(columns=[c for c in table.columns if c.startswith("pred_")]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
