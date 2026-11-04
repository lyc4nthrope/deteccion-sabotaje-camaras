"""Tests for the synthetic sabotage generator (T18)."""

import cv2
import numpy as np
import pandas as pd

from src import synthetic_tampering as S


def _scene(seed=0, shape=(180, 320, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_sabotage_all_returns_the_four_classes_with_the_same_shape():
    image = _scene(1)
    out = S.sabotage_all(image, seed=5)
    assert set(out) == {"desenfocada", "girada", "tapada_total", "tapada_parcial"}
    for variant in out.values():
        assert variant.shape == image.shape and variant.dtype == np.uint8


def test_same_seed_gives_the_same_variants():
    image = _scene(2)
    a = S.sabotage_all(image, seed=9)
    b = S.sabotage_all(image, seed=9)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_different_seed_changes_the_variants():
    image = _scene(3)
    a = S.sabotage_all(image, seed=1)
    b = S.sabotage_all(image, seed=2)
    assert not np.array_equal(a["tapada_total"], b["tapada_total"])


def test_blur_lowers_the_sharpness():
    image = _scene(4)
    blurred = S.blur(image, np.random.default_rng(0))
    lap = lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert lap(blurred) < lap(image) / 5


def test_rotation_changes_a_textured_image():
    image = _scene(8)
    assert not np.array_equal(S.rotate(image, np.random.default_rng(0)), image)


def test_rotation_leaves_no_black_corners():
    image = np.full((180, 320, 3), 150, np.uint8)
    rotated = S.rotate(image, np.random.default_rng(0))
    assert rotated.min() > 0  # reflection fills the corners, so no pure black borders


def test_total_occlusion_is_one_uniform_color():
    out = S.occlude_total(_scene(5), np.random.default_rng(0))
    assert (out == out[0, 0]).all()


def test_partial_occlusion_covers_the_configured_share():
    image = np.full((180, 320, 3), 200, np.uint8)
    out = S.occlude_partial(image, np.random.default_rng(0))
    changed = (out != 200).any(axis=-1).mean()
    assert S.PARTIAL_AREA_RANGE[0] - 0.05 <= changed <= S.PARTIAL_AREA_RANGE[1] + 0.05


def test_generate_writes_four_classes_per_image(tmp_path):
    images = tmp_path / "img"
    images.mkdir()
    names = []
    for i in range(3):
        name = f"imagen1_2023010{i}_000000.jpg"
        cv2.imwrite(str(images / name), _scene(i))
        names.append(name)
    listing = tmp_path / "list.csv"
    pd.DataFrame({"archivo": names}).to_csv(listing, index=False)

    out = tmp_path / "out"
    assert S.generate(listing, images, out, seed=0) == 12
    for label in ("desenfocada", "girada", "tapada_total", "tapada_parcial"):
        assert len(list((out / label).glob("*.jpg"))) == 3


def test_generate_is_reproducible_from_the_same_seed(tmp_path):
    images = tmp_path / "img"
    images.mkdir()
    name = "imagen1_20230101_000000.jpg"
    cv2.imwrite(str(images / name), _scene(7))
    listing = tmp_path / "list.csv"
    pd.DataFrame({"archivo": [name]}).to_csv(listing, index=False)
    S.generate(listing, images, tmp_path / "a", seed=3)
    S.generate(listing, images, tmp_path / "b", seed=3)
    for label in ("desenfocada", "girada", "tapada_total", "tapada_parcial"):
        a = cv2.imread(str(tmp_path / "a" / label / "imagen1_20230101_000000.jpg"))
        b = cv2.imread(str(tmp_path / "b" / label / "imagen1_20230101_000000.jpg"))
        assert np.array_equal(a, b)
