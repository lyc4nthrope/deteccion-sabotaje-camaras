"""Compute the descriptor vector of every image once and cache it as CSV (T35).

Run from the repository root:

    python -m src.build_features

Location 1 (train/val/test, purged images excluded) uses the 8 references of data/splits/referencias.csv.
Location 2 (unseen camera) is calibrated with 8 of its own normal images, which are excluded from the output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import data, features

OUT_DIR = data.RESULTS_DIR / "clasico"


def _matcher(paths: pd.Series, root: Path) -> features.ReferenceMatcher:
    return features.ReferenceMatcher([features.preprocess(features.load_image(root / p))["gray"] for p in paths])


def describe(images: pd.DataFrame, matcher: features.ReferenceMatcher, root: Path = data.BREMEN_ROOT) -> pd.DataFrame:
    """Descriptor vector of each image, keeping its path, label and set."""
    rows = []
    for path, label, subset in zip(images["path"], images["label"], images["set"]):
        vector = features.extract(features.load_image(root / path), matcher)
        rows.append({"path": path, "label": label, "set": subset, **vector})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = data.load_split()
    split = split[split["set"] != data.PURGED]
    references = pd.read_csv(data.SPLITS_DIR / "referencias.csv")
    location1 = describe(split, _matcher(references["path"], data.BREMEN_ROOT))
    location1.to_csv(OUT_DIR / "descriptores_ubicacion1.csv", index=False)

    unseen = data.load_split(data.SPLITS_DIR / "camara_no_vista.csv")
    calibration, evaluation = data.calibration_split(unseen)
    calibration.to_csv(data.SPLITS_DIR / "referencias_camara_no_vista.csv", index=False)
    location2 = describe(evaluation, _matcher(calibration["path"], data.BREMEN_ROOT))
    location2.to_csv(OUT_DIR / "descriptores_camara_no_vista.csv", index=False)

    print(location1.groupby(["label", "set"]).size().unstack(fill_value=0).to_string())
    print(f"Cámara no vista: {len(location2)} imágenes (sin sus {len(calibration)} referencias)")


if __name__ == "__main__":
    main()
