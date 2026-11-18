"""Live demo on the laptop (T82, T84, T85): webcam or video -> classification with FPS on screen.

T82, classic module: the first CALIBRATION_FRAMES frames of the live scene are the normal references for ORB
(the classic module needs references of the same camera, T33). Then every frame is classified with the thresholds
calibrated on train (results/clasico/umbrales.json, T41).

T84, optional CNN: --cnn <run folder in results/cnn> adds the CNN prediction on the same frame.

T85, correction keys: c = CLAHE, g = inverse gamma (level 2), m = median. Active filters are applied to the frame
before classification; the screen shows the prediction before and after the filters, with FPS.
Keys: q quits.

Run: python -m src.demo --source 0            (webcam)
     python -m src.demo --source video.mp4    (recorded video)
     python -m src.demo --source video.mp4 --headless   (no window: prints the labels and FPS, for tests)
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import classic, corrections, features

ROOT = Path(__file__).resolve().parents[1]
CLASSIC = ROOT / "results" / "clasico"
CALIBRATION_FRAMES = 8  # as in the article (T82)
# A webcam starts with a default exposure and corrects it over its first frames (measured: mean gray 74 -> 103 in
# about 2 s). References taken in that window do not match the settled scene, so ORB matches drop and RANSAC reports
# false rotations. About 1.5 s at 20 fps are discarded first. A video file is already exposed: nothing to discard.
WARMUP_FRAMES = 30
FILTER_KEYS = {"c": "clahe", "g": "gamma_inverso_nivel_2", "m": "mediana"}


def warmup_frames(source) -> int:
    """Frames to discard before calibrating: only a live camera (a numeric source) needs to settle its exposure."""
    return WARMUP_FRAMES if str(source).isdigit() else 0


def read_calibration(capture, warmup: int) -> list[np.ndarray]:
    """Discard `warmup` frames, then return the next CALIBRATION_FRAMES frames as references."""
    frames = []
    for _ in range(warmup):
        ok, _frame = capture.read()
        if not ok:
            raise RuntimeError("source ended during the camera warm-up")
    while len(frames) < CALIBRATION_FRAMES:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("source ended before the calibration frames")
        frames.append(frame)
    return frames


def calibrate(frames: list[np.ndarray]) -> features.ReferenceMatcher:
    """ORB references of the live scene from the calibration frames (T82)."""
    return features.ReferenceMatcher([features.preprocess(frame)["gray"] for frame in frames])


def apply_filters(frame: np.ndarray, active: set[str]) -> np.ndarray:
    """Apply the active correction filters in a fixed order (T91)."""
    out = frame
    for key in ("c", "g", "m"):
        if key in active:
            out = corrections.CORRECTIONS[FILTER_KEYS[key]](out)
    return out


def classify_classic(
    frame: np.ndarray, matcher: features.ReferenceMatcher, thresholds: classic.Thresholds
) -> tuple[str, dict]:
    """Label of the decision tree and the descriptors it used, so the demo can show why it decided."""
    desc = features.extract(frame, matcher)
    return str(classic.predict(pd.DataFrame([desc]), thresholds)[0]), desc


def descriptor_lines(desc: dict, thresholds: classic.Thresholds, gray_mean: float) -> list[str]:
    """On-screen diagnosis: each descriptor of the decision tree next to its threshold (value / threshold).

    ASCII only, because cv2.putText cannot draw accents or the degree sign.
    """
    angle = desc["angle"]
    angle_text = "sin angulo" if np.isnan(angle) else f"angulo {abs(angle):.2f} / {thresholds.angle_max:.2f}"
    cover = desc[thresholds.cover_feature]
    return [
        f"ORB {int(desc['orb_matches'])} / {thresholds.match_min:.1f}   {angle_text}",
        f"Laplaciano {desc['lap_var']:.1f} / {thresholds.blur_max:.1f}   matiz {cover:.1f} / "
        f"{thresholds.cover_threshold:.1f}   gris {gray_mean:.0f}",
    ]


def classify_cnn(frame: np.ndarray, model) -> str:
    from src import cnn

    rgb = cv2.cvtColor(cv2.resize(frame, (cnn.IMAGE_SIZE, cnn.IMAGE_SIZE), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    batch = rgb[None].astype(np.float32) / 127.5 - 1.0
    return cnn.CLASSES[int(np.argmax(model.predict(batch, verbose=0)[0]))]


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    for i, text in enumerate(lines):
        origin = (10, 25 + 24 * i)
        # Black outline first, so the yellow text reads on a bright wall as well as on a dark scene.
        cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def run(source, cnn_run: str | None = None, headless: bool = False, max_frames: int | None = None) -> dict:
    thresholds = classic.load_thresholds(CLASSIC / "umbrales.json")
    model = None
    if cnn_run:
        import tensorflow as tf

        model = tf.keras.models.load_model(ROOT / "results" / "cnn" / cnn_run / "model.keras")

    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source {source!r}")

    matcher = calibrate(read_calibration(capture, warmup_frames(source)))

    active: set[str] = set()
    counts: Counter = Counter()
    fps = 0.0
    frames = 0
    last = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames is not None and frames >= max_frames):
            break
        frames += 1
        filtered = apply_filters(frame, active)
        before, desc = classify_classic(frame, matcher, thresholds)
        if active:
            after, desc = classify_classic(filtered, matcher, thresholds)
        else:
            after = before
        counts[after] += 1
        gray_mean = float(cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).mean())
        lines = [f"clasico: {before}" + (f" -> con filtros: {after}" if active else ""), f"FPS: {fps:.1f}"]
        lines += descriptor_lines(desc, thresholds, gray_mean)  # values of the image the classic decided on
        if model is not None:
            lines.append(f"CNN: {classify_cnn(filtered, model)}")
        if active:
            lines.append("filtros: " + ",".join(sorted(active)))
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        if headless:
            continue
        cv2.imshow("demo", overlay(filtered, lines))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if chr(key) in FILTER_KEYS:
            name = chr(key)
            active.symmetric_difference_update({name})

    capture.release()
    if not headless:
        cv2.destroyAllWindows()
    return {"frames": frames, "labels": dict(counts), "fps": round(fps, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0")
    parser.add_argument("--cnn", default=None, help="folder in results/cnn with model.keras (T84)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    print(run(args.source, args.cnn, args.headless, args.max_frames))


if __name__ == "__main__":
    main()
