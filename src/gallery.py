"""Gallery of Bremen images: 5 per class and location (T12).

Samples come from data/splits/split.csv, which has no duplicates (D6), with a fixed seed. Output:
results/figuras/galeria.png, one row per location and class, five columns.
Run: python -m src.gallery
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import cnn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figuras" / "galeria.png"
PER_CELL = 5
THUMB = (200, 113)  # 16:9 thumbnails, as the 1280x720 originals


def build(seed: int = 0) -> np.ndarray:
    table = pd.read_csv(cnn.SPLIT)
    rows = []
    for location in sorted(table["location"].unique()):
        for label in cnn.CLASSES:
            cell = table[(table["location"] == location) & (table["label"] == label)]
            take = cell.sample(min(PER_CELL, len(cell)), random_state=seed)
            images = []
            for path in take["path"]:
                image = cv2.imread(str(cnn.BREMEN_ROOT / path))
                images.append(cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA))
            while len(images) < PER_CELL:
                images.append(np.zeros((THUMB[1], THUMB[0], 3), np.uint8))
            row = np.hstack(images)
            label_bar = np.full((22, row.shape[1], 3), 40, np.uint8)
            cv2.putText(label_bar, f"ubicacion {location} - {label} ({len(cell)})", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            rows.append(np.vstack([label_bar, row]))
    return np.vstack(rows)


def main() -> None:
    sheet = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet)
    print(f"wrote {OUT} {sheet.shape}")


if __name__ == "__main__":
    main()
