"""Blur people before an image of the own capture is stored (T86, Ley 1581 de 2012).

A pretrained detector (YOLOv8n, class 17) finds the COCO class "person"; every box, enlarged by a margin, is
replaced by a strong blur. The detector is a parameter, so the tests use a fake one and need no weights.

Install the detector apart from the main requirements (it pulls PyTorch; ultralytics is AGPL-3.0, fine for
academic use):  pip install -r requirements-anonymize.txt
Run:  python -m src.anonymize <raw folder> <output folder>
Weights and photos are never committed (.gitignore).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

PERSON_CLASS = 0  # "person" in COCO
MIN_CONFIDENCE = 0.25  # ultralytics default: a low value blurs more, which is the safe side for privacy
MARGIN = 0.10  # share of the box size added on each side, so hair and shoulders are covered too
MOSAIC_FACTOR = 16  # the box is shrunk this many times and enlarged back: a strong blur at any box size
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
Detector = Callable[[np.ndarray], Sequence[Box]]


def _blur_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape[:2]
    small = cv2.resize(region, (max(1, w // MOSAIC_FACTOR), max(1, h // MOSAIC_FACTOR)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (5, 5), 0)


def blur_boxes(image: np.ndarray, boxes: Iterable[Box], margin: float = MARGIN) -> np.ndarray:
    """Return a copy of the image with every box, enlarged by the margin and clipped to the image, blurred."""
    out = image.copy()
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        dx, dy = round((x2 - x1) * margin), round((y2 - y1) * margin)
        x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
        x2, y2 = min(width, x2 + dx), min(height, y2 + dy)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = _blur_region(out[y1:y2, x1:x2])
    return out


def anonymize(image: np.ndarray, detector: Detector) -> tuple[np.ndarray, int]:
    """Blur every person the detector finds; also return how many there were."""
    boxes = list(detector(image))
    return blur_boxes(image, boxes), len(boxes)


def anonymize_folder(source: Path | str, target: Path | str, detector: Detector) -> dict[str, int]:
    """Write a blurred copy of every image of source into target. Return people blurred per file name."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("target must differ from source: the raw images are kept until they are checked")
    target.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot read image: {path}")
        blurred, report[path.name] = anonymize(image, detector)
        cv2.imwrite(str(target / path.name), blurred)
    return report


class YoloPersonDetector:
    """People boxes from a pretrained YOLOv8n (downloaded once by ultralytics on first use)."""

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = MIN_CONFIDENCE):
        from ultralytics import YOLO  # optional dependency, see requirements-anonymize.txt

        self.model = YOLO(weights)
        self.confidence = confidence

    def __call__(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        result = self.model(image, classes=[PERSON_CLASS], conf=self.confidence, verbose=False)[0]
        boxes = []
        for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy():
            box = (max(0, int(x1)), max(0, int(y1)), min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2))))
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="folder with the raw images")
    parser.add_argument("target", help="folder for the blurred copies")
    args = parser.parse_args()
    report = anonymize_folder(args.source, args.target, YoloPersonDetector())
    print(f"{len(report)} images, {sum(report.values())} people blurred")
    for name, count in report.items():
        if count:
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
