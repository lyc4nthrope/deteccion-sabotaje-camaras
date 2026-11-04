"""CNN data loading and model (T60, T61).

Uses the Bremen split (T14, data/splits/split.csv). Only the train, val and test sets are used; the "descartada"
rows are the guard band between sets (D6) and never enter training or evaluation. Images are 224x224 RGB, scaled to
[-1, 1] with the MobileNetV2 preprocessing. The four classes follow D1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BREMEN_ROOT = ROOT / "data" / "raw" / "bremen"
SPLIT = ROOT / "data" / "splits" / "split.csv"

IMAGE_SIZE = 224
CLASSES = ("normal", "desenfocada", "girada", "tapada")  # D1: four classes


def read_split(split_csv: Path = SPLIT) -> pd.DataFrame:
    """The train, val and test rows of the split, with an integer label column (y)."""
    table = pd.read_csv(split_csv)
    table = table[table["set"].isin(["train", "val", "test"])].copy()
    unknown = set(table["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown labels in split: {sorted(unknown)}")
    table["y"] = table["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return table.reset_index(drop=True)


def load_images(paths: list[str], root: Path = BREMEN_ROOT) -> np.ndarray:
    """Read the images as RGB at IMAGE_SIZE, scaled to [-1, 1]."""
    import cv2

    batch = np.empty((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for i, rel in enumerate(paths):
        image = cv2.imread(str(root / rel))
        if image is None:
            raise FileNotFoundError(f"cannot read {rel} under {root}")
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        batch[i] = image[..., ::-1]  # BGR -> RGB
    return batch / 127.5 - 1.0


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Balanced weights: n_samples / (n_classes * count). Compensates the small "girada" class."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    weights = len(labels) / (len(CLASSES) * np.maximum(counts, 1.0))
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(trainable_layers: int = 0, learning_rate: float = 1e-3, weights: str | None = "imagenet"):
    """MobileNetV2 with a 4-class head (T61).

    trainable_layers: how many of the last layers of the base are unfrozen (0 = frozen base).
    weights="imagenet" downloads the pretrained weights; pass None for tests without network.
    """
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=weights, pooling="avg"
    )
    base.trainable = False
    if trainable_layers > 0:
        base.trainable = True
        for layer in base.layers[:-trainable_layers]:
            layer.trainable = False
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
