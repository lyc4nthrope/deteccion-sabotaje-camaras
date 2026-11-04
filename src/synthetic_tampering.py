"""Synthetic sabotage for the Medellín cross test (T18, D12).

Medellín has no sabotage, so each normal image produces three sabotaged versions, as in the UHCTD dataset:
- desenfocada: Gaussian blur
- girada: rotation with a crop back to the original size (the corners are filled by reflection, never black)
- tapada: total occlusion (uniform cover) or partial occlusion (one rectangle of 30-60 % of the area)

All random choices come from one seed, so the same seed regenerates the same set.
Run: python -m src.synthetic_tampering --list data/splits/medellin_normales.csv --out data/medellin_sintetico
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BLUR_SIGMA = (4.0, 8.0)  # Gaussian sigma range, in pixels (1280x720 originals)
ROTATION_RANGE = (10.0, 45.0)  # degrees, sign chosen at random
PARTIAL_AREA_RANGE = (0.30, 0.60)  # share of the image covered by the partial occlusion


def blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(*BLUR_SIGMA))
    return cv2.GaussianBlur(image, (0, 0), sigma)


def rotate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(*ROTATION_RANGE)) * (1 if rng.random() < 0.5 else -1)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def occlude_total(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    color = rng.integers(0, 256, 3)
    return np.full_like(image, color, dtype=np.uint8)


def occlude_partial(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    area = float(rng.uniform(*PARTIAL_AREA_RANGE))
    box_h = int(round(h * np.sqrt(area)))
    box_w = int(round(w * np.sqrt(area)))
    top = int(rng.integers(0, h - box_h + 1))
    left = int(rng.integers(0, w - box_w + 1))
    out = image.copy()
    out[top:top + box_h, left:left + box_w] = rng.integers(0, 256, 3)
    return out


def sabotage_all(image: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """The four sabotage variants of one normal image, all from the same seeded generator."""
    rng = np.random.default_rng(seed)
    return {
        "desenfocada": blur(image, rng),
        "girada": rotate(image, rng),
        "tapada_total": occlude_total(image, rng),
        "tapada_parcial": occlude_partial(image, rng),
    }


def generate(list_csv: Path, images_root: Path, out_dir: Path, seed: int = 0) -> int:
    """Write the sabotaged images under out_dir/<class>/. Returns the number of files written."""
    files = pd.read_csv(list_csv)["archivo"].tolist()
    written = 0
    for index, name in enumerate(files):
        image = cv2.imread(str(images_root / name))
        if image is None:
            raise FileNotFoundError(f"cannot read {name} under {images_root}")
        for label, variant in sabotage_all(image, seed + index).items():
            target = out_dir / label
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f"{Path(name).stem}.jpg"), variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, default=Path("data/splits/medellin_normales.csv"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/medellin/images"))
    parser.add_argument("--out", type=Path, default=Path("data/medellin_sintetico"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    count = generate(args.list, args.images, args.out, args.seed)
    print(f"wrote {count} sabotaged images to {args.out}")


if __name__ == "__main__":
    main()
