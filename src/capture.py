"""Helper for the own capture in Armenia (T81, protocol in docs/captura.md).

Takes short bursts with a fixed webcam, names every file as the protocol says
(`AAAAMMDD_HHMM_<clase>_<luz>_<n>.jpg`), keeps one folder per class and appends a row to the session sheet.
Afterwards it can blur people in every class folder (src/anonymize.py). Photos are never committed: `data/` is ignored.

Capture a burst (move nothing in the scene between bursts of the same class and light):
    python -m src.capture capturar --clase tapada --luz contraluz --n 50 --intervalo 2 \\
        --lugar "Armenia, entrada" --camara "webcam 720p" --soporte "1,5 m, 10 grados"
Blur the people of every class folder into data/propios_anonimas/ (raw images are kept):
    python -m src.capture anonimizar
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from src import anonymize, data

CLASSES = ("normal", "desenfocada", "girada", "tapada")
LIGHTS = ("apagada", "lampara", "contraluz")
ROOT = data.ROOT / "data" / "propios"
BLURRED_ROOT = data.ROOT / "data" / "propios_anonimas"
SHEET = ROOT / "hoja_sesion.csv"
SHEET_COLUMNS = ["fecha", "hora", "lugar", "camara", "resolucion", "soporte", "luz", "clase", "imagenes",
                 "observaciones"]
WARM_UP_FRAMES = 10  # the webcam adjusts exposure and focus during its first frames


def _check(clase: str, luz: str) -> None:
    if clase not in CLASSES:
        raise ValueError(f"unknown class {clase!r}; use one of {CLASSES}")
    if luz not in LIGHTS:
        raise ValueError(f"unknown light {luz!r}; use one of {LIGHTS}")


def file_name(when: datetime, clase: str, luz: str, n: int) -> str:
    _check(clase, luz)
    return f"{when:%Y%m%d_%H%M}_{clase}_{luz}_{n:03d}.jpg"


def next_index(folder: Path, clase: str, luz: str) -> int:
    """1 + the highest number already used for this class and light in the folder."""
    pattern = re.compile(rf"_{clase}_{luz}_(\d{{3}})\.jpg$")
    used = [int(m.group(1)) for p in Path(folder).glob("*.jpg") if (m := pattern.search(p.name))] if Path(folder).exists() else []
    return max(used, default=0) + 1


def capture_burst(
    camera,
    root: Path,
    clase: str,
    luz: str,
    n: int,
    interval: float,
    now: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Save n frames of the camera in root/<clase>/, one every `interval` seconds. Return how many were saved.

    Numbering continues after the files already there, so a second burst never overwrites the first. If the camera
    stops answering, the error is raised and what was saved before it is kept.
    """
    _check(clase, luz)
    folder = Path(root) / clase
    folder.mkdir(parents=True, exist_ok=True)
    start = next_index(folder, clase, luz)
    for i in range(n):
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError(f"the camera stopped answering after {i} of {n} frames")
        if not cv2.imwrite(str(folder / file_name(now(), clase, luz, start + i)), frame):
            raise RuntimeError("cannot write the image to disk")
        if i < n - 1:
            sleep(interval)
    return n


def log_session(sheet: Path, when: datetime, clase: str, luz: str, count: int, resolution: tuple[int, int],
                meta: dict) -> None:
    """Append one row to the session sheet (created with its header the first time)."""
    sheet = Path(sheet)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    new = not sheet.exists()
    row = {
        "fecha": f"{when:%Y-%m-%d}",
        "hora": f"{when:%H:%M}",
        "lugar": meta.get("lugar", ""),
        "camara": meta.get("camara", ""),
        "resolucion": f"{resolution[0]}x{resolution[1]}",
        "soporte": meta.get("soporte", ""),
        "luz": luz,
        "clase": clase,
        "imagenes": count,
        "observaciones": meta.get("observaciones", ""),
    }
    with open(sheet, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def anonymize_all(raw_root: Path, target_root: Path, detector) -> dict[str, dict[str, int]]:
    """Blur the people of every class folder of raw_root into target_root/<class>/ (raw images are kept)."""
    report = {}
    for clase in CLASSES:
        source = Path(raw_root) / clase
        if source.is_dir():
            report[clase] = anonymize.anonymize_folder(source, Path(target_root) / clase, detector)
    return report


def _open_camera(source: str):
    camera = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not camera.isOpened():
        raise SystemExit(f"cannot open the camera {source!r}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(WARM_UP_FRAMES):
        camera.read()
    return camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capturar", help="take a burst of images of one class and light")
    cap.add_argument("--clase", required=True, choices=CLASSES)
    cap.add_argument("--luz", required=True, choices=LIGHTS)
    cap.add_argument("--n", type=int, default=50, help="images in the burst (about 100 per class in total)")
    cap.add_argument("--intervalo", type=float, default=2.0, help="seconds between images")
    cap.add_argument("--espera", type=int, default=5, help="seconds before the first image, to get out of the frame")
    cap.add_argument("--fuente", default="0", help="camera index or video path")
    for field in ("lugar", "camara", "soporte", "observaciones"):
        cap.add_argument(f"--{field}", default="")

    sub.add_parser("anonimizar", help="blur the people of every class folder into data/propios_anonimas/")
    args = parser.parse_args()

    if args.command == "anonimizar":
        report = anonymize_all(ROOT, BLURRED_ROOT, anonymize.YoloPersonDetector())
        for clase, files in report.items():
            print(f"{clase}: {len(files)} images, {sum(files.values())} people blurred")
        return

    camera = _open_camera(args.fuente)
    try:
        ok, frame = camera.read()
        resolution = (frame.shape[1], frame.shape[0]) if ok else (0, 0)
        for left in range(args.espera, 0, -1):
            print(f"starting in {left} s...", flush=True)
            time.sleep(1)
        started = datetime.now()
        count = capture_burst(camera, ROOT, args.clase, args.luz, args.n, args.intervalo)
    finally:
        camera.release()
    log_session(SHEET, started, args.clase, args.luz, count, resolution, vars(args))
    print(f"{count} images in {ROOT / args.clase}; session row added to {SHEET}")
    print("Check that nobody identifiable appears, then run: python -m src.capture anonimizar")


if __name__ == "__main__":
    main()
