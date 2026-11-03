"""Dataset inventory, deduplication, train/val/test split and reference selection.

Tasks T11, T13, T14, T15 and T19 of docs/PLAN.md. Run from the repository root:

    python -m src.data

It reads data/raw/bremen/ and writes results/inventario.csv and the files in data/splits/.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BREMEN_ROOT = ROOT / "data" / "raw" / "bremen"
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"

# Dataset folder suffix -> project class name.
LABELS = {"Normal": "normal", "Blur": "desenfocada", "Rotated": "girada", "Noimage": "tapada"}
FOLDER_RE = re.compile(r"^Location(\d+)_(Normal|Blur|Rotated|Noimage)$")
TIMESTAMP_RE = re.compile(r"image_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{1,2})")
COPY_MARKERS = (" - Copy", "(")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 42
DEFAULT_BLOCK_SIZE = 12
DEFAULT_REFERENCES = 8


def parse_timestamp(name: str) -> datetime:
    """Return the capture time encoded in a file name like image_2024-07-18_19-44-19.jpg."""
    match = TIMESTAMP_RE.search(name)
    if match is None:
        raise ValueError(f"File name without a timestamp: {name}")
    return datetime(*(int(part) for part in match.groups()))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan_images(root: Path = BREMEN_ROOT) -> pd.DataFrame:
    """List every image of the class folders with its location, label, capture time and content hash."""
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        match = FOLDER_RE.match(folder.name)
        if match is None:
            continue
        location, suffix = int(match.group(1)), match.group(2)
        for image in sorted(folder.glob("*.jpg")):
            rows.append(
                {
                    "path": image.relative_to(root).as_posix(),
                    "location": location,
                    "label": LABELS[suffix],
                    "timestamp": parse_timestamp(image.name),
                    "md5": _md5(image),
                }
            )
    return pd.DataFrame(rows, columns=["path", "location", "label", "timestamp", "md5"])


def _is_copy(path: str) -> bool:
    name = Path(path).name
    return any(marker in name for marker in COPY_MARKERS)


def deduplicate(records: pd.DataFrame) -> pd.DataFrame:
    """Keep one file per image content.

    Inside a folder, the original name wins over " - Copy" and "(1)" copies. An image whose content also
    appears in a lower location is dropped from the higher one: Location 2 "Noimage" is a copy of Location 1
    and would otherwise leak training images into the unseen-camera test.
    """
    ranked = records.assign(_copy=records["path"].map(_is_copy), _len=records["path"].str.len())
    ranked = ranked.sort_values(["location", "_copy", "_len", "path"])
    unique = ranked.drop_duplicates(subset="md5", keep="first")
    return unique.drop(columns=["_copy", "_len"]).sort_values("path").reset_index(drop=True)


def build_inventory(records: pd.DataFrame, unique: pd.DataFrame) -> pd.DataFrame:
    """Count files and distinct images per location and class."""
    files = records.groupby(["location", "label"]).size().rename("files")
    kept = unique.groupby(["location", "label"]).size().rename("unique")
    inventory = pd.concat([files, kept], axis=1).fillna(0).astype(int).reset_index()
    inventory["duplicates"] = inventory["files"] - inventory["unique"]
    return inventory


def _time_blocks(records: pd.DataFrame, block_size: int) -> pd.Series:
    """Group each class into blocks of block_size consecutive images in time.

    Normal images are one minute apart and the covered-lens ones come in bursts of one per second, so a
    fixed number of consecutive images adapts to both: about 12 minutes or about 12 seconds per block.
    """
    ordered = records.sort_values(["label", "timestamp", "path"])
    position = ordered.groupby("label").cumcount()
    blocks = ordered["label"] + "_" + (position // block_size).map("{:04d}".format)
    return blocks.reindex(records.index)


def make_split(unique: pd.DataFrame, seed: int = DEFAULT_SEED, block_size: int = DEFAULT_BLOCK_SIZE) -> pd.DataFrame:
    """Split Location 1 into train/val/test by blocks of consecutive images, separately for each class.

    Consecutive images are almost identical, so a whole block goes to the same set. Each class keeps at
    least one block in val and in test when it has three or more blocks.
    """
    rng = np.random.default_rng(seed)
    split = unique[unique["location"] == 1].copy()
    split["block"] = _time_blocks(split, block_size)
    split["set"] = ""
    for label in sorted(split["label"].unique()):
        blocks = sorted(split.loc[split["label"] == label, "block"].unique())
        rng.shuffle(blocks)
        n = len(blocks)
        n_val = max(1, round(n * SPLIT_RATIOS["val"])) if n >= 3 else 0
        n_test = max(1, round(n * SPLIT_RATIOS["test"])) if n >= 3 else 0
        assignment = {b: "val" for b in blocks[:n_val]}
        assignment |= {b: "test" for b in blocks[n_val : n_val + n_test]}
        assignment |= {b: "train" for b in blocks[n_val + n_test :]}
        mask = split["label"] == label
        split.loc[mask, "set"] = split.loc[mask, "block"].map(assignment)
    return split.sort_values("path").reset_index(drop=True)


def unseen_camera_set(unique: pd.DataFrame) -> pd.DataFrame:
    """Location 2 images, never used for training or calibration (T19)."""
    unseen = unique[unique["location"] == 2].copy()
    unseen["block"] = _time_blocks(unseen, DEFAULT_BLOCK_SIZE)
    unseen["set"] = "camara_no_vista"
    return unseen.sort_values("path").reset_index(drop=True)


def select_references(split: pd.DataFrame, n: int = DEFAULT_REFERENCES, subset: str = "train") -> pd.DataFrame:
    """Pick n normal images of one subset spread over the capture times (different light conditions).

    Use subset="train" for Location 1 and subset="camara_no_vista" to calibrate on Location 2; in the latter
    case the chosen references must be excluded from the evaluation.
    """
    candidates = split[(split["label"] == "normal") & (split["set"] == subset)]
    candidates = candidates.sort_values("timestamp").reset_index(drop=True)
    if len(candidates) <= n:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, n).round().astype(int)
    return candidates.iloc[positions].reset_index(drop=True)


def load_split(path: Path = SPLITS_DIR / "split.csv") -> pd.DataFrame:
    """Read a split file written by this module."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def main() -> None:
    records = scan_images(BREMEN_ROOT)
    unique = deduplicate(records)
    inventory = build_inventory(records, unique)
    split = make_split(unique)
    unseen = unseen_camera_set(unique)
    references = select_references(split)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(RESULTS_DIR / "inventario.csv", index=False)
    split.to_csv(SPLITS_DIR / "split.csv", index=False)
    unseen.to_csv(SPLITS_DIR / "camara_no_vista.csv", index=False)
    references.to_csv(SPLITS_DIR / "referencias.csv", index=False)

    print(inventory.to_string(index=False))
    print("\nDivisión (ubicación 1):")
    print(split.groupby(["label", "set"]).size().unstack(fill_value=0).to_string())
    print(f"\nCámara no vista (ubicación 2): {len(unseen)} imágenes")
    print(f"Referencias: {len(references)}")


if __name__ == "__main__":
    main()
