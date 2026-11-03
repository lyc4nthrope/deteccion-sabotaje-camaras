"""Tests for the dataset inventory, deduplication, split and references (T11, T13, T14, T15, T19)."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import data

BREMEN_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "bremen"


def _write_image(path: Path, seed: int) -> None:
    """Write a small JPEG whose content depends only on the seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """Small dataset: 2 locations x 4 classes, 1 image per minute, 6 hours per class."""
    seed = 0
    for location in ("Location1", "Location2"):
        for folder, day in (("Normal", 16), ("Blur", 18), ("Rotated", 20), ("Noimage", 22)):
            if location == "Location2" and folder == "Noimage":
                continue
            for hour in range(6, 12):
                for minute in (0, 20, 40):
                    name = f"image_2024-07-{day}_{hour:02d}-{minute:02d}-00.jpg"
                    _write_image(tmp_path / f"{location}_{folder}" / name, seed)
                    seed += 1
    # Exact copies inside a folder, with the suffixes seen in the real dataset.
    original = tmp_path / "Location1_Rotated" / "image_2024-07-20_06-00-00.jpg"
    _copy(original, tmp_path / "Location1_Rotated" / "image_2024-07-20_06-00-00 - Copy.jpg")
    _copy(original, tmp_path / "Location1_Rotated" / "image_2024-07-20_06-00-00 (1).jpg")
    # Location 2 "Noimage" folder that is a copy of Location 1 images (as in the real dataset).
    for src in sorted((tmp_path / "Location1_Noimage").glob("*.jpg"))[:5]:
        _copy(src, tmp_path / "Location2_Noimage" / src.name)
    return tmp_path


# ---------------------------------------------------------------- parse_timestamp


def test_parse_timestamp_reads_date_and_time():
    assert data.parse_timestamp("image_2024-07-18_19-44-19.jpg") == datetime(2024, 7, 18, 19, 44, 19)


@pytest.mark.parametrize(
    "name",
    ["image_2024-08-24_19-03-14 - Copy.jpg", "image_2024-08-24_19-03-14 (1).jpg", "image_2024-08-24_19-03-14(1).jpg"],
)
def test_parse_timestamp_accepts_copy_suffixes(name):
    assert data.parse_timestamp(name) == datetime(2024, 8, 24, 19, 3, 14)


def test_parse_timestamp_accepts_truncated_seconds():
    # Real file in Location1_Rotated: "image_2024-08-24_19-06-3 (1).jpg".
    assert data.parse_timestamp("image_2024-08-24_19-06-3 (1).jpg") == datetime(2024, 8, 24, 19, 6, 3)


def test_parse_timestamp_rejects_unknown_names():
    with pytest.raises(ValueError):
        data.parse_timestamp("foto.jpg")


# ---------------------------------------------------------------- scan / dedupe


def test_scan_maps_folders_to_locations_and_labels(fake_root):
    records = data.scan_images(fake_root)
    labels = set(zip(records["location"], records["label"]))
    assert (1, "normal") in labels and (1, "desenfocada") in labels
    assert (1, "girada") in labels and (1, "tapada") in labels
    assert (2, "normal") in labels


def test_scan_ignores_non_class_folders(fake_root):
    _write_image(fake_root / "zips" / "image_2024-07-16_06-00-00.jpg", 999)
    records = data.scan_images(fake_root)
    assert not records["path"].str.startswith("zips").any()


def test_deduplicate_keeps_one_file_per_content_and_the_original_name(fake_root):
    records = data.scan_images(fake_root)
    unique = data.deduplicate(records)
    rotated = unique[(unique["location"] == 1) & (unique["label"] == "girada")]
    assert len(rotated) == 18
    assert not unique["md5"].duplicated().any()
    kept = unique.loc[unique["md5"] == records.loc[records["path"].str.contains("Copy"), "md5"].iloc[0], "path"]
    assert kept.iloc[0].endswith("image_2024-07-20_06-00-00.jpg")


def test_deduplicate_drops_location2_copies_of_location1(fake_root):
    unique = data.deduplicate(data.scan_images(fake_root))
    assert unique[(unique["location"] == 2) & (unique["label"] == "tapada")].empty
    assert len(unique[(unique["location"] == 1) & (unique["label"] == "tapada")]) == 18


def test_inventory_counts_files_and_unique_images(fake_root):
    records = data.scan_images(fake_root)
    inventory = data.build_inventory(records, data.deduplicate(records))
    row = inventory[(inventory["location"] == 1) & (inventory["label"] == "girada")].iloc[0]
    assert row["files"] == 20 and row["unique"] == 18
    row = inventory[(inventory["location"] == 2) & (inventory["label"] == "tapada")].iloc[0]
    assert row["files"] == 5 and row["unique"] == 0


# ---------------------------------------------------------------- split


def _split(fake_root, seed=42):
    unique = data.deduplicate(data.scan_images(fake_root))
    return data.make_split(unique, seed=seed, block_size=3)


def test_split_uses_only_location1_and_assigns_every_image(fake_root):
    unique = data.deduplicate(data.scan_images(fake_root))
    split = data.make_split(unique, seed=42, block_size=3)
    assert set(split["location"]) == {1}
    assert len(split) == (unique["location"] == 1).sum()
    assert set(split["set"]) == {"train", "val", "test"}


def test_split_never_puts_one_hour_block_in_two_sets(fake_root):
    split = _split(fake_root)
    sets_per_block = split.groupby(["label", "block"])["set"].nunique()
    assert (sets_per_block == 1).all()


def test_split_blocks_are_consecutive_images_in_time(fake_root):
    unique = data.deduplicate(data.scan_images(fake_root))
    split = data.make_split(unique, seed=42, block_size=4)
    sizes = split.groupby(["label", "block"]).size()
    assert sizes.max() == 4
    for _, block in split.groupby(["label", "block"]):
        times = block.sort_values("timestamp")
        others = split[(split["label"] == block["label"].iloc[0]) & (split["block"] != block["block"].iloc[0])]
        inside = others[(others["timestamp"] > times["timestamp"].iloc[0]) & (others["timestamp"] < times["timestamp"].iloc[-1])]
        assert inside.empty, "a block must not interleave with another block of the same class"


def test_split_has_every_class_in_every_set_with_roughly_70_15_15(fake_root):
    split = _split(fake_root)
    for label, group in split.groupby("label"):
        counts = group["set"].value_counts()
        assert set(counts.index) == {"train", "val", "test"}, label
        assert counts["train"] > counts["val"] and counts["train"] > counts["test"], label


def test_split_is_reproducible_and_depends_on_the_seed(fake_root):
    a = _split(fake_root, seed=1).sort_values("path").reset_index(drop=True)
    b = _split(fake_root, seed=1).sort_values("path").reset_index(drop=True)
    c = _split(fake_root, seed=2).sort_values("path").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert not a["set"].equals(c["set"])


# ---------------------------------------------------------------- unseen camera / references


def test_unseen_camera_has_location2_without_location1_content(fake_root):
    unique = data.deduplicate(data.scan_images(fake_root))
    split = data.make_split(unique, seed=42, block_size=3)
    unseen = data.unseen_camera_set(unique)
    assert set(unseen["location"]) == {2}
    assert set(unseen["label"]) == {"normal", "desenfocada", "girada"}
    assert not set(unseen["md5"]) & set(split["md5"])


def test_references_are_normal_train_images_spread_over_the_day(fake_root):
    split = _split(fake_root)
    refs = data.select_references(split, n=4)
    assert len(refs) == 4
    assert set(refs["label"]) == {"normal"} and set(refs["set"]) == {"train"}
    hours = sorted(pd.to_datetime(refs["timestamp"]).dt.hour)
    assert hours[-1] - hours[0] >= 2


def test_references_return_all_available_when_fewer_than_n(fake_root):
    split = _split(fake_root)
    available = ((split["label"] == "normal") & (split["set"] == "train")).sum()
    assert len(data.select_references(split, n=available + 10)) == available


def test_load_split_round_trip(fake_root, tmp_path):
    split = _split(fake_root)
    out = tmp_path / "split.csv"
    split.to_csv(out, index=False)
    loaded = data.load_split(out)
    assert len(loaded) == len(split)
    assert set(loaded.columns) >= {"path", "location", "label", "set", "timestamp", "md5", "block"}


# ---------------------------------------------------------------- real dataset


@pytest.mark.skipif(not BREMEN_ROOT.exists(), reason="Bremen dataset not downloaded (see data/README.md)")
def test_real_dataset_counts():
    records = data.scan_images(BREMEN_ROOT)
    unique = data.deduplicate(records)
    files = records.groupby(["location", "label"]).size().to_dict()
    assert files == {
        (1, "normal"): 1800, (1, "desenfocada"): 600, (1, "girada"): 600, (1, "tapada"): 600,
        (2, "normal"): 753, (2, "desenfocada"): 260, (2, "girada"): 167, (2, "tapada"): 251,
    }
    kept = unique.groupby(["location", "label"]).size().to_dict()
    assert kept == {
        (1, "normal"): 1774, (1, "desenfocada"): 600, (1, "girada"): 135, (1, "tapada"): 600,
        (2, "normal"): 753, (2, "desenfocada"): 251, (2, "girada"): 167,
    }


@pytest.mark.skipif(not BREMEN_ROOT.exists(), reason="Bremen dataset not downloaded (see data/README.md)")
def test_real_split_is_balanced_for_every_class():
    unique = data.deduplicate(data.scan_images(BREMEN_ROOT))
    split = data.make_split(unique)
    for label, group in split.groupby("label"):
        share = group["set"].value_counts(normalize=True)
        assert 0.60 <= share["train"] <= 0.80, (label, share.to_dict())
        assert 0.08 <= share["val"] <= 0.25 and 0.08 <= share["test"] <= 0.25, (label, share.to_dict())


def test_references_can_be_taken_from_the_unseen_camera(fake_root):
    unseen = data.unseen_camera_set(data.deduplicate(data.scan_images(fake_root)))
    refs = data.select_references(unseen, n=3, subset="camara_no_vista")
    assert len(refs) == 3
    assert set(refs["location"]) == {2} and set(refs["label"]) == {"normal"}
