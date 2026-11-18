"""Tests for the live demo (T82, T84, T85), without a camera: video files and headless runs."""

import cv2
import numpy as np
import pytest

from src import demo


def _scene(seed=0, shape=(360, 640, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_apply_filters_without_keys_returns_the_frame():
    frame = _scene(1)
    assert np.array_equal(demo.apply_filters(frame, set()), frame)


@pytest.mark.parametrize("key", ["c", "g", "m"])
def test_each_filter_key_changes_the_frame(key):
    frame = _scene(2)
    assert not np.array_equal(demo.apply_filters(frame, {key}), frame)


def test_filters_keep_shape_and_dtype_together():
    frame = _scene(3)
    out = demo.apply_filters(frame, {"c", "g", "m"})
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_unknown_filter_key_is_ignored():
    frame = _scene(4)
    assert np.array_equal(demo.apply_filters(frame, {"x"}), frame)


def test_calibrate_builds_a_matcher_from_the_frames():
    frames = [_scene(s) for s in range(demo.CALIBRATION_FRAMES)]
    matcher = demo.calibrate(frames)
    assert len(matcher.references) == demo.CALIBRATION_FRAMES


def _write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_headless_run_counts_frames_after_calibration(tmp_path):
    video = tmp_path / "clip.avi"
    frames = [_scene(i) for i in range(demo.CALIBRATION_FRAMES + 5)]
    _write_video(video, frames)
    result = demo.run(str(video), headless=True)
    assert result["frames"] == 5
    assert sum(result["labels"].values()) == 5


def test_headless_run_fails_clearly_on_a_missing_source(tmp_path):
    with pytest.raises(RuntimeError):
        demo.run(str(tmp_path / "missing.avi"), headless=True)


class _FakeCamera:
    """Gives frames 0, 1, 2, ... where every pixel of frame i equals i, like a capture that never ends."""

    def __init__(self):
        self.count = 0

    def read(self):
        frame = np.full((360, 640, 3), self.count % 256, dtype=np.uint8)
        self.count += 1
        return True, frame


def test_a_camera_discards_warmup_frames_and_a_video_does_not():
    assert demo.warmup_frames("0") == demo.WARMUP_FRAMES > 0
    assert demo.warmup_frames("clip.avi") == 0


_THRESHOLDS = demo.classic.Thresholds(
    match_min=12.5, blur_max=18.86, cover_feature="hue_std", cover_threshold=5.40, cover_direction="<", angle_max=0.94
)


def test_descriptor_lines_show_each_value_next_to_its_threshold():
    desc = {"orb_matches": 652, "angle": 0.013, "lap_var": 6.31, "hue_std": 34.07}
    text = " ".join(demo.descriptor_lines(desc, _THRESHOLDS, gray_mean=108.4))
    for expected in ("ORB 652 / 12.5", "angulo 0.01 / 0.94", "Laplaciano 6.3 / 18.9", "matiz 34.1 / 5.4", "gris 108"):
        assert expected in text


def test_descriptor_lines_say_when_there_is_no_reliable_angle():
    desc = {"orb_matches": 0, "angle": float("nan"), "lap_var": 4.2, "hue_std": 5.0}
    assert "sin angulo" in " ".join(demo.descriptor_lines(desc, _THRESHOLDS, gray_mean=17.0))


def test_classify_classic_returns_the_descriptors_it_decided_with():
    frames = [_scene(s) for s in range(demo.CALIBRATION_FRAMES)]
    label, desc = demo.classify_classic(frames[0], demo.calibrate(frames), _THRESHOLDS)
    assert label in ("normal", "desenfocada", "girada", "tapada")
    assert {"orb_matches", "angle", "lap_var", "hue_std"} <= set(desc)


def test_calibration_skips_the_warmup_frames():
    camera = _FakeCamera()
    frames = demo.read_calibration(camera, warmup=30)
    assert len(frames) == demo.CALIBRATION_FRAMES
    assert [int(f[0, 0, 0]) for f in frames] == list(range(30, 30 + demo.CALIBRATION_FRAMES))
