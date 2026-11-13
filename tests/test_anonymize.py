"""Tests for blurring people before saving the own capture (T86, Ley 1581 de 2012)."""

import importlib.util

import cv2
import numpy as np
import pytest

from src import anonymize as A


def _image(seed=0, size=(360, 640)):
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


class FakeDetector:
    """Returns fixed boxes, so the tests do not need YOLO weights."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return self.boxes


# ---------------------------------------------------------------- blur_boxes


def test_image_without_people_is_returned_identical():
    image = _image()
    assert np.array_equal(A.blur_boxes(image, []), image)


def test_every_pixel_inside_a_box_is_changed_and_outside_is_untouched():
    image = _image(1)
    out = A.blur_boxes(image, [(100, 50, 200, 250)], margin=0.0)
    inside = (out[50:250, 100:200] != image[50:250, 100:200]).any(axis=2)
    assert inside.mean() > 0.95
    outside = np.ones(image.shape[:2], bool)
    outside[50:250, 100:200] = False
    assert np.array_equal(out[outside], image[outside])


def test_blur_removes_the_detail_inside_the_box():
    # The Laplacian variance measures detail: a strong blur must remove almost all of it.
    image = _image(2)
    out = A.blur_boxes(image, [(0, 0, 640, 360)], margin=0.0)
    detail = lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert detail(out) < detail(image) / 100


def test_blur_is_strong_enough_to_hide_a_face_sized_pattern():
    # 32 px of diagonal stripes 4 px wide stand for the detail of a face. After the blur its contrast
    # (standard deviation) must almost vanish; a mild blur would keep the pattern readable.
    pattern = (np.indices((32, 32)).sum(axis=0) // 4 % 2 * 255).astype(np.uint8)
    image = np.zeros((360, 640, 3), np.uint8)
    image[100:132, 100:132] = pattern[..., None]
    out = A.blur_boxes(image, [(100, 100, 132, 132)], margin=0.0)
    assert out[100:132, 100:132].std() < 0.1 * image[100:132, 100:132].std()


def test_margin_enlarges_the_blurred_region():
    image = _image(3)
    out = A.blur_boxes(image, [(200, 100, 300, 200)], margin=0.2)
    # 20 % of the 100 px box = 20 px on each side: pixel (185, 150) is now inside the blurred area.
    assert (out[150, 185] != image[150, 185]).any()
    assert np.array_equal(out[150, 170], image[150, 170])


def test_boxes_outside_the_image_are_clipped():
    image = _image(4)
    out = A.blur_boxes(image, [(-50, -50, 100, 100), (600, 300, 900, 500)], margin=0.0)
    assert out.shape == image.shape
    assert (out[0:100, 0:100] != image[0:100, 0:100]).any()


def test_empty_or_inverted_boxes_are_ignored():
    image = _image(5)
    assert np.array_equal(A.blur_boxes(image, [(100, 100, 100, 200), (300, 300, 250, 250)], margin=0.0), image)


def test_input_image_is_not_modified():
    image = _image(6)
    copy = image.copy()
    A.blur_boxes(image, [(0, 0, 320, 180)])
    assert np.array_equal(image, copy)


# ---------------------------------------------------------------- anonymize and folders


def test_anonymize_reports_how_many_people_were_blurred():
    detector = FakeDetector([(10, 10, 60, 120), (300, 40, 380, 300)])
    out, count = A.anonymize(_image(7), detector)
    assert count == 2 and detector.calls == 1 and out.shape == (360, 640, 3)


def test_anonymize_folder_writes_every_image_and_never_overwrites_the_source(tmp_path):
    src, dst = tmp_path / "crudas", tmp_path / "anonimas"
    src.mkdir()
    for name in ("a.jpg", "b.png"):
        cv2.imwrite(str(src / name), _image(8))
    (src / "notas.txt").write_text("no es imagen")
    report = A.anonymize_folder(src, dst, FakeDetector([(0, 0, 50, 50)]))
    assert sorted(report) == ["a.jpg", "b.png"] and all(n == 1 for n in report.values())
    assert sorted(p.name for p in dst.iterdir()) == ["a.jpg", "b.png"]
    with pytest.raises(ValueError):
        A.anonymize_folder(src, src, FakeDetector([]))


def test_person_class_is_the_coco_person():
    assert A.PERSON_CLASS == 0


# ---------------------------------------------------------------- real detector (optional)


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="pip install -r requirements-anonymize.txt")
def test_yolo_detector_returns_boxes_inside_the_image():
    detector = A.YoloPersonDetector()
    boxes = detector(_image(9))
    assert all(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360 for x1, y1, x2, y2 in boxes)
