"""Synthetic sabotage for the Medellín cross test (T18, D12).

Medellín has no sabotage, so each normal image produces four sabotaged versions, aimed at the Bremen classes:
- desenfocada: Gaussian blur.
- girada: a pan, as in Bremen, where the camera points at another part of the scene. A window of 60-80 % of the
  frame is cut at one edge and resized back to the original size. No reflection borders are created.
- tapada_total: uniform cover in a realistic color (dark, gray, skin or red), with a gentle gradient and light noise.
- tapada_parcial: one rectangle covering 85-95 % of the frame, so the visible rest is too small to count as a
  normal scene. It stays in the "tapada" class, as the classic module would call it.

Every image and variant gets its own seed (SeedSequence), so changing one variant does not move the others.
A manifest.csv records file, group, label and parameters of every output.
Run: python -m src.synthetic_tampering --list data/splits/medellin_normales.csv --out data/medellin_sintetico
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BLUR_SIGMA = (4.0, 8.0)  # Gaussian sigma range, in pixels (1280x720 originals)
PAN_WINDOW_RANGE = (0.60, 0.80)  # width and height of the pan window, as a share of the frame
PARTIAL_AREA_RANGE = (0.85, 0.95)  # share of the frame covered by the partial occlusion
# Cover colors (BGR) seen in real tampering: dark cloth, gray, skin, red cloth.
COVER_PALETTE = [(35, 35, 35), (110, 110, 110), (140, 160, 210), (40, 40, 170)]
LABELS = ("desenfocada", "girada", "tapada_total", "tapada_parcial")


def blur(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    sigma = float(rng.uniform(*BLUR_SIGMA))
    return cv2.GaussianBlur(image, (0, 0), sigma), {"sigma": round(sigma, 2)}


def pan(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Camera pan: crop a window at one edge and resize it back to the original size."""
    h, w = image.shape[:2]
    share = float(rng.uniform(*PAN_WINDOW_RANGE))
    win_w, win_h = int(round(w * share)), int(round(h * share))
    side = "left" if rng.random() < 0.5 else "right"
    x0 = 0 if side == "left" else w - win_w
    y0 = int(rng.integers(0, h - win_h + 1))
    crop = image[y0:y0 + win_h, x0:x0 + win_w]
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return out, {"ventana": round(share, 3), "lado": side, "y0": y0}


def occlude_total(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    base = np.array(COVER_PALETTE[int(rng.integers(len(COVER_PALETTE)))], np.float32)
    tilt = float(rng.uniform(-20, 20))
    ramp = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None, None] * tilt
    noise = rng.normal(0.0, 4.0, (h, w, 3)).astype(np.float32)
    out = np.clip(base + ramp + noise, 0, 255).astype(np.uint8)
    return out, {"color_bgr": base.astype(int).tolist(), "degradado": round(tilt, 1)}


def occlude_partial(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    area = float(rng.uniform(*PARTIAL_AREA_RANGE))
    box_h, box_w = int(round(h * np.sqrt(area))), int(round(w * np.sqrt(area)))
    top = int(rng.integers(0, h - box_h + 1))
    left = int(rng.integers(0, w - box_w + 1))
    out = image.copy()
    color = np.array(COVER_PALETTE[int(rng.integers(len(COVER_PALETTE)))], np.uint8)
    out[top:top + box_h, left:left + box_w] = color
    return out, {"area": round(area, 3), "top": top, "left": left}


def sabotage_all(image: np.ndarray, seed: int) -> dict[str, tuple[np.ndarray, dict]]:
    """The four variants of one image. Each variant draws from its own child of the seed sequence."""
    children = np.random.SeedSequence(seed).spawn(4)
    makers = {
        "desenfocada": blur,
        "girada": pan,
        "tapada_total": occlude_total,
        "tapada_parcial": occlude_partial,
    }
    return {label: maker(image, np.random.default_rng(child))
            for (label, maker), child in zip(makers.items(), children)}


def generate(list_csv: Path, images_root: Path, out_dir: Path, seed: int = 0) -> int:
    """Write the sabotaged images under out_dir/<label>/ and out_dir/manifest.csv. Returns the number written."""
    listing = pd.read_csv(list_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["archivo", "grupo", "etiqueta", "parametros"])
        for index, row in enumerate(listing.itertuples(index=False)):
            name = row.archivo
            image = cv2.imread(str(images_root / name))
            if image is None:
                raise FileNotFoundError(f"cannot read {name} under {images_root}")
            for label, (variant, params) in sabotage_all(image, [seed, index]).items():
                target = out_dir / label
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target / f"{Path(name).stem}.jpg"), variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
                writer.writerow([name, row.grupo, label, json.dumps(params)])
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
    print(f"wrote {count} sabotaged images and manifest to {args.out}")


if __name__ == "__main__":
    main()
