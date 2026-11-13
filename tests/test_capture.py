"""Tests for the own-capture helper (T81). The camera and the clock are injected: no webcam is needed."""

from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

from src import capture as C

WHEN = datetime(2026, 11, 4, 18, 30)


class FakeCamera:
    """Returns numbered frames (the first pixel holds the frame number) and counts the reads."""

    def __init__(self, fail_after=None):
        self.reads, self.fail_after = 0, fail_after

    def read(self):
        if self.fail_after is not None and self.reads >= self.fail_after:
            return False, None
        self.reads += 1
        frame = np.zeros((72, 128, 3), np.uint8)
        frame[0, 0] = self.reads
        return True, frame


# ---------------------------------------------------------------- names


def test_file_name_follows_the_protocol():
    assert C.file_name(WHEN, "tapada", "contraluz", 12) == "20261104_1830_tapada_contraluz_012.jpg"


@pytest.mark.parametrize("clase,luz", [("otra", "lampara"), ("normal", "sol")])
def test_unknown_class_or_light_is_rejected(clase, luz):
    with pytest.raises(ValueError):
        C.file_name(WHEN, clase, luz, 1)


def test_numbering_continues_after_the_existing_files(tmp_path):
    folder = tmp_path / "tapada"
    folder.mkdir()
    for n in (1, 2, 7):
        (folder / f"20261104_1830_tapada_contraluz_{n:03d}.jpg").write_bytes(b"x")
    (folder / "20261104_1830_tapada_lampara_099.jpg").write_bytes(b"x")  # another light: not counted
    assert C.next_index(folder, "tapada", "contraluz") == 8
    assert C.next_index(tmp_path / "nueva", "tapada", "contraluz") == 1


# ---------------------------------------------------------------- burst


def test_burst_saves_n_numbered_frames_in_the_class_folder(tmp_path):
    sleeps = []
    saved = C.capture_burst(FakeCamera(), tmp_path, "normal", "lampara", n=3, interval=2.0, now=lambda: WHEN,
                            sleep=sleeps.append)
    folder = tmp_path / "normal"
    assert sorted(p.name for p in folder.iterdir()) == [
        "20261104_1830_normal_lampara_001.jpg",
        "20261104_1830_normal_lampara_002.jpg",
        "20261104_1830_normal_lampara_003.jpg",
    ]
    assert saved == 3 and sleeps == [2.0, 2.0]  # no wait after the last frame


def test_a_second_burst_does_not_overwrite_the_first(tmp_path):
    C.capture_burst(FakeCamera(), tmp_path, "girada", "apagada", n=2, interval=0, now=lambda: WHEN, sleep=lambda s: None)
    C.capture_burst(FakeCamera(), tmp_path, "girada", "apagada", n=2, interval=0, now=lambda: WHEN, sleep=lambda s: None)
    names = sorted(p.name for p in (tmp_path / "girada").iterdir())
    assert [n[-7:-4] for n in names] == ["001", "002", "003", "004"]


def test_burst_stops_with_an_error_when_the_camera_stops_answering(tmp_path):
    with pytest.raises(RuntimeError):
        C.capture_burst(FakeCamera(fail_after=2), tmp_path, "tapada", "apagada", n=5, interval=0, now=lambda: WHEN,
                        sleep=lambda s: None)
    assert len(list((tmp_path / "tapada").iterdir())) == 2  # what was captured before the failure is kept


def test_saved_files_are_readable_jpegs(tmp_path):
    C.capture_burst(FakeCamera(), tmp_path, "normal", "apagada", n=1, interval=0, now=lambda: WHEN, sleep=lambda s: None)
    image = cv2.imread(str(next((tmp_path / "normal").iterdir())))
    assert image is not None and image.shape == (72, 128, 3)


# ---------------------------------------------------------------- session sheet


def test_session_row_is_appended_with_the_protocol_fields(tmp_path):
    sheet = tmp_path / "hoja_sesion.csv"
    meta = {"lugar": "Armenia", "camara": "webcam", "soporte": "1,5 m, 10 grados", "observaciones": ""}
    C.log_session(sheet, WHEN, "tapada", "contraluz", 100, (1280, 720), meta)
    C.log_session(sheet, WHEN, "normal", "lampara", 50, (1280, 720), meta)
    table = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    assert list(table.columns) == ["fecha", "hora", "lugar", "camara", "resolucion", "soporte", "luz", "clase",
                                   "imagenes", "observaciones"]
    assert table.loc[0, ["fecha", "hora", "clase", "luz", "imagenes", "resolucion"]].tolist() == [
        "2026-11-04", "18:30", "tapada", "contraluz", "100", "1280x720"]
    assert len(table) == 2


# ---------------------------------------------------------------- anonymization of every class folder


def test_anonymize_all_writes_a_copy_per_class_and_keeps_the_raw_folder(tmp_path):
    raw, out = tmp_path / "propios", tmp_path / "anonimas"
    for clase in ("normal", "tapada"):
        (raw / clase).mkdir(parents=True)
        cv2.imwrite(str(raw / clase / f"{clase}.png"), np.full((90, 160, 3), 120, np.uint8))
    report = C.anonymize_all(raw, out, detector=lambda image: [(10, 10, 60, 60)])
    assert report == {"normal": {"normal.png": 1}, "tapada": {"tapada.png": 1}}
    assert (raw / "normal" / "normal.png").exists() and (out / "tapada" / "tapada.png").exists()
