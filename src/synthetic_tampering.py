"""Synthetic sabotage for the Medellín cross test (T18, D12).

Medellín has no sabotage, so each normal image produces four sabotaged versions, aimed at the Bremen classes. The
parameters were chosen by measuring them with the classic module on Bremen images, so that the descriptors of the
synthetic images fall in the range of the real ones (tests/test_synthetic_realism.py):
- desenfocada: Gaussian blur with sigma 8 to 14 pixels of the 1280x720 original (4 to 7 at the 640x360 working
  size, where it removes the keypoints), plus a little noise (the real blurred images keep a sensor noise floor).
- girada: a pan, as in Bremen, where the camera points at another part of the scene. The view is **displaced** by
  65-90 % of the frame width (so at most 35 % of the original view stays) and rotated 0.5 to 6 degrees, as the
  estimated angles of the real pans. It is not a zoom. The exposed area is filled by reflection: it has texture
  but its keypoints do not match the reference, as a part of the scene never seen before would not.
- tapada_total: uniform cover in a realistic color (dark, gray, skin or red), with a gentle gradient and noise at the sensor floor
  (real covers have a Laplacian variance of about 0.9 to 6.5).
- tapada_parcial: one rectangle covering 85-95 % of the frame, with the same texture as the total cover (a flat
  rectangle next to a noisy total cover would be a trivial cue), so the visible rest is too small to count as a
  normal scene. It stays in the "tapada" class, as the classic module would call it.

Every image and variant gets its own seed: the seed of the image comes from the file name (CRC32), not from its
position in the list, so reordering the list changes nothing; each variant draws from its own child of that seed.
A manifest.csv records file, group, label and parameters of every output.
Run: python -m src.synthetic_tampering --list data/splits/medellin_normales.csv --out data/medellin_sintetico
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BLUR_SIGMA = (8.0, 14.0)  # Gaussian sigma range, in pixels (1280x720 originals)
NOISE_SIGMA = (0.5, 0.9)  # sensor noise left after the blur, in gray levels (real blurred images have a floor)
COVER_NOISE = (0.3, 0.9)  # noise of a cover, in gray levels: real covers have a Laplacian variance of 0.9 to 6.5
PAN_SHIFT_RANGE = (0.65, 0.90)  # horizontal displacement of the view, as a share of the frame width
PAN_ROTATION_RANGE = (0.5, 6.0)  # in-plane rotation of the pan, in degrees (either sign)
PARTIAL_AREA_RANGE = (0.85, 0.95)  # share of the frame covered by the partial occlusion
# Cover colors (BGR) seen in real tampering: dark cloth, gray, skin, red cloth.
COVER_PALETTE = [(35, 35, 35), (110, 110, 110), (140, 160, 210), (40, 40, 170)]
LABELS = ("desenfocada", "girada", "tapada_total", "tapada_parcial")


def blur(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    sigma = float(rng.uniform(*BLUR_SIGMA))
    noise = float(rng.uniform(*NOISE_SIGMA))
    blurred = cv2.GaussianBlur(image, (0, 0), sigma).astype(np.float32)
    out = np.clip(blurred + rng.normal(0.0, noise, image.shape), 0, 255).round().astype(np.uint8)
    return out, {"sigma": round(sigma, 2), "ruido": round(noise, 2)}


def pan(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Camera pan: the view is displaced horizontally and rotated a little; the exposed area is reflected."""
    h, w = image.shape[:2]
    share = round(float(rng.uniform(*PAN_SHIFT_RANGE)), 3)
    pixels = int(round(share * w))
    side = "left" if rng.random() < 0.5 else "right"
    angle = round(float(rng.uniform(*PAN_ROTATION_RANGE)) * (1 if rng.random() < 0.5 else -1), 2)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[0, 2] += pixels if side == "left" else -pixels  # the camera turns left: the content moves right
    out = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out, {"lado": side, "desplazamiento": share, "pixeles": pixels, "rotacion": angle}


def cover(height: int, width: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A cover of one realistic color with a gentle vertical gradient and light noise (total and partial covers)."""
    base = np.array(COVER_PALETTE[int(rng.integers(len(COVER_PALETTE)))], np.float32)
    tilt = float(rng.uniform(-20, 20))
    ramp = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None] * tilt
    sigma = float(rng.uniform(*COVER_NOISE))
    noise = rng.normal(0.0, sigma, (height, width, 3)).astype(np.float32)
    out = np.clip(base + ramp + noise, 0, 255).round().astype(np.uint8)
    return out, {"color_bgr": base.astype(int).tolist(), "degradado": round(tilt, 1), "ruido": round(sigma, 2)}


def occlude_total(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    return cover(h, w, rng)


def occlude_partial(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    area = float(rng.uniform(*PARTIAL_AREA_RANGE))
    box_h, box_w = int(round(h * np.sqrt(area))), int(round(w * np.sqrt(area)))
    top = int(rng.integers(0, h - box_h + 1))
    left = int(rng.integers(0, w - box_w + 1))
    out = image.copy()
    patch, params = cover(box_h, box_w, rng)
    out[top:top + box_h, left:left + box_w] = patch
    return out, {"area": round(area, 3), "top": top, "left": left, **params}


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


def file_seed(seed: int, name: str) -> list[int]:
    """Seed of one image: from its file name, so reordering or filtering the list does not change any variant."""
    return [seed, zlib.crc32(name.encode("utf-8"))]


def generate(list_csv: Path, images_root: Path, out_dir: Path, seed: int = 0) -> int:
    """Write the sabotaged images under out_dir/<label>/ and out_dir/manifest.csv. Returns the number written."""
    listing = pd.read_csv(list_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["archivo", "grupo", "etiqueta", "parametros"])
        for row in listing.itertuples(index=False):
            name = row.archivo
            image = cv2.imread(str(images_root / name))
            if image is None:
                raise FileNotFoundError(f"cannot read {name} under {images_root}")
            for label, (variant, params) in sabotage_all(image, file_seed(seed, name)).items():
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
