"""Tests for the synthetic sabotage generator (T18)."""

import csv
import json

import cv2
import numpy as np
import pandas as pd

from src import synthetic_tampering as S


def _scene(seed=0, shape=(180, 320, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _texture_gray(seed=0):
    return cv2.cvtColor(_scene(seed, shape=(180, 320, 3)), cv2.COLOR_BGR2GRAY)


def _lap(img):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------------------------------------------------------- contract


def test_sabotage_all_returns_the_four_classes_with_shape_and_params():
    image = _scene(1)
    out = S.sabotage_all(image, seed=5)
    assert set(out) == set(S.LABELS)
    for variant, params in out.values():
        assert variant.shape == image.shape and variant.dtype == np.uint8
        assert isinstance(params, dict) and params


def test_same_seed_gives_the_same_variants_and_params():
    image = _scene(2)
    a = S.sabotage_all(image, seed=9)
    b = S.sabotage_all(image, seed=9)
    for label in S.LABELS:
        assert np.array_equal(a[label][0], b[label][0]) and a[label][1] == b[label][1]


def test_changing_the_seed_changes_every_variant():
    image = _scene(3)
    a = S.sabotage_all(image, seed=1)
    b = S.sabotage_all(image, seed=2)
    assert all(not np.array_equal(a[k][0], b[k][0]) for k in S.LABELS)


def test_each_variant_draws_from_its_own_child_seed():
    # The blur variant uses the first child of the image seed, whatever the other variants draw.
    image = _scene(4)
    rng_a = np.random.default_rng(np.random.SeedSequence([0, 0]).spawn(4)[0])
    blur_a, _ = S.blur(image, rng_a)
    blur_b, _ = S.sabotage_all(image, seed=[0, 0])["desenfocada"]
    assert np.array_equal(blur_a, blur_b)


# ---------------------------------------------------------------- blur


def test_blur_lowers_the_sharpness():
    image = _scene(5)
    blurred, _ = S.blur(image, np.random.default_rng(0))
    assert _lap(blurred) < _lap(image) / 5


def test_blur_is_strong_enough_for_the_640x360_working_size_and_keeps_a_sensor_noise_floor():
    # sigma is in pixels of the 1280x720 original: halved by the reduction to 640x360 it must still remove the
    # keypoints (a sigma of 4 to 8 left the classic module calling most blurred images "normal").
    assert S.BLUR_SIGMA[0] >= 8.0
    _, params = S.blur(_scene(5), np.random.default_rng(1))
    assert S.BLUR_SIGMA[0] <= params["sigma"] <= S.BLUR_SIGMA[1]
    assert S.NOISE_SIGMA[0] <= params["ruido"] <= S.NOISE_SIGMA[1]


# ---------------------------------------------------------------- pan (girada)


def test_pan_params_describe_a_displacement_with_little_overlap_and_a_small_rotation():
    image = _scene(6, shape=(360, 640, 3))
    out, params = S.pan(image, np.random.default_rng(0))
    assert out.shape == image.shape
    assert params["lado"] in {"left", "right"}
    assert S.PAN_SHIFT_RANGE[0] <= params["desplazamiento"] <= S.PAN_SHIFT_RANGE[1]
    assert S.PAN_ROTATION_RANGE[0] <= abs(params["rotacion"]) <= S.PAN_ROTATION_RANGE[1]
    assert 1 - params["desplazamiento"] <= 0.35  # at most 35 % of the original view stays in the frame


def test_pan_is_a_displacement_not_a_zoom(monkeypatch):
    """With no rotation the visible part of the original is copied unscaled, shifted by the stated amount."""
    monkeypatch.setattr(S, "PAN_ROTATION_RANGE", (0.0, 0.0))
    image = _scene(6, shape=(360, 640, 3))
    for seed in range(6):
        out, params = S.pan(image, np.random.default_rng(seed))
        shift = params["pixeles"]
        assert shift == round(params["desplazamiento"] * 640)
        if params["lado"] == "left":  # the camera turns left: the content moves right
            assert np.array_equal(out[:, shift:], image[:, : 640 - shift])
        else:
            assert np.array_equal(out[:, : 640 - shift], image[:, shift:])


def test_pan_creates_no_black_borders():
    # A uniform frame stays uniform after a pan: the exposed area is filled by reflection, never with black.
    image = np.full((360, 640, 3), 150, np.uint8)
    out, _ = S.pan(image, np.random.default_rng(0))
    assert np.array_equal(out, image)


def test_pan_changes_a_textured_scene():
    image = _scene(7, shape=(360, 640, 3))
    out, _ = S.pan(image, np.random.default_rng(3))
    assert out.shape == image.shape and not np.array_equal(out, image)


# ---------------------------------------------------------------- tapada


def test_total_occlusion_is_nearly_uniform_and_uses_a_realistic_color():
    image = _scene(8)
    out, params = S.occlude_total(image, np.random.default_rng(0))
    assert tuple(params["color_bgr"]) in {tuple(map(int, c)) for c in S.COVER_PALETTE}
    # Gradient and light noise only: per channel the spread stays small, so it is still a cover, not a scene.
    assert out.reshape(-1, 3).std(axis=0).max() < 20


def test_partial_occlusion_covers_at_least_85_percent_of_the_frame():
    image = np.full((180, 320, 3), 200, np.uint8)
    out, params = S.occlude_partial(image, np.random.default_rng(0))
    assert S.PARTIAL_AREA_RANGE[0] <= params["area"] <= S.PARTIAL_AREA_RANGE[1]
    assert S.PARTIAL_AREA_RANGE[0] >= 0.85
    visible = (out == 200).all(axis=-1).mean()
    assert visible <= 1 - S.PARTIAL_AREA_RANGE[0] + 0.02


def test_cover_noise_stays_at_the_sensor_floor_of_real_covers():
    # Real covers have a Laplacian variance of about 0.9 to 6.5. Noise of 4 gray levels gave 38, above the
    # sharpness threshold, so the classic module called every synthetic cover "girada".
    out, params = S.occlude_total(_scene(8), np.random.default_rng(0))
    assert S.COVER_NOISE[0] <= params["ruido"] <= S.COVER_NOISE[1] and S.COVER_NOISE[1] <= 1.0
    assert _lap(out) < 18.0


def test_partial_occlusion_has_the_same_texture_as_the_total_cover():
    # A flat rectangle next to a noisy total cover would be a trivial cue: both come from the same function.
    image = np.full((180, 320, 3), 200, np.uint8)
    out, params = S.occlude_partial(image, np.random.default_rng(0))
    top, left = params["top"], params["left"]
    box = out[top:top + 120, left:left + 200].astype(float)
    assert np.diff(box, axis=1).std() > 0.4  # the noise is there, the rectangle is not a flat color
    assert {"color_bgr", "degradado", "ruido"} <= set(params)
    assert tuple(params["color_bgr"]) in {tuple(map(int, c)) for c in S.COVER_PALETTE}


# ---------------------------------------------------------------- generate


def _write_fixture(tmp_path, names):
    images = tmp_path / "img"
    images.mkdir()
    for i, name in enumerate(names):
        cv2.imwrite(str(images / name), _scene(i, shape=(120, 160, 3)))
    listing = tmp_path / "list.csv"
    pd.DataFrame({"archivo": names, "grupo": [1] * len(names)}).to_csv(listing, index=False)
    return images, listing


def test_generate_writes_four_classes_per_image_and_a_manifest(tmp_path):
    names = [f"imagen1_2023010{i}_000000.jpg" for i in range(3)]
    images, listing = _write_fixture(tmp_path, names)
    out = tmp_path / "out"
    assert S.generate(listing, images, out, seed=0) == 12
    for label in S.LABELS:
        assert len(list((out / label).glob("*.jpg"))) == 3
    with (out / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12
    assert {r["etiqueta"] for r in rows} == set(S.LABELS)
    assert all(json.loads(r["parametros"]) for r in rows)


def test_generate_is_reproducible_from_the_same_seed(tmp_path):
    name = "imagen1_20230101_000000.jpg"
    images, listing = _write_fixture(tmp_path, [name])
    S.generate(listing, images, tmp_path / "a", seed=3)
    S.generate(listing, images, tmp_path / "b", seed=3)
    for label in S.LABELS:
        a = cv2.imread(str(tmp_path / "a" / label / "imagen1_20230101_000000.jpg"))
        b = cv2.imread(str(tmp_path / "b" / label / "imagen1_20230101_000000.jpg"))
        assert np.array_equal(a, b)


def test_file_seed_depends_on_the_file_name_and_not_on_its_position():
    assert S.file_seed(0, "a.jpg") == S.file_seed(0, "a.jpg")
    assert S.file_seed(0, "a.jpg") != S.file_seed(0, "b.jpg")
    assert S.file_seed(0, "a.jpg") != S.file_seed(1, "a.jpg")


def test_reordering_the_list_does_not_change_any_variant(tmp_path):
    names = [f"imagen1_2023010{i}_000000.jpg" for i in range(3)]
    images, listing = _write_fixture(tmp_path, names)
    S.generate(listing, images, tmp_path / "a", seed=0)
    pd.read_csv(listing).iloc[::-1].to_csv(listing, index=False)
    S.generate(listing, images, tmp_path / "b", seed=0)
    for label in S.LABELS:
        for name in names:
            a = cv2.imread(str(tmp_path / "a" / label / name))
            b = cv2.imread(str(tmp_path / "b" / label / name))
            assert np.array_equal(a, b)
