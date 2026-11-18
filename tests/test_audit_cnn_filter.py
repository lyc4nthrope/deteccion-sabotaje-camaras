"""Tests for the CNN audit with a correction applied before the network (T90, T93)."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src import audit, audit_cnn as AC, corrections
from src import audit_cnn_filter as F

PLAN = [("sal_pimienta", 3), ("jpeg", 3)]


class FixedModel:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=np.float32)
        self.batches = []

    def predict(self, batch, verbose=0):
        self.batches.append(np.asarray(batch))
        return np.tile(self.proba, (len(batch), 1))


def _images(tmp_path, n=2):
    rows = []
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"i{i}.png"), np.random.default_rng(i).integers(0, 256, (720, 1280, 3), dtype=np.uint8))
        rows.append({"path": f"i{i}.png", "label": "normal" if i == 0 else "tapada"})
    return pd.DataFrame(rows)


def test_the_correction_is_applied_to_the_degraded_frame_before_the_network(tmp_path):
    images = _images(tmp_path, n=1)
    model = FixedModel([0.7, 0.1, 0.1, 0.1])
    F.filtered_predictions(model, images, PLAN, "mediana", root=tmp_path)
    original = cv2.imread(str(tmp_path / "i0.png"))
    for i, (name, level) in enumerate(PLAN):
        degraded = audit.degrade(original, name, level, "i0.png")
        expected = AC.to_cnn_input(corrections.CORRECTIONS["mediana"](degraded))
        np.testing.assert_array_equal(model.batches[0][i], expected)


def test_a_filtered_frame_is_not_the_unfiltered_one(tmp_path):
    images = _images(tmp_path, n=1)
    model = FixedModel([0.7, 0.1, 0.1, 0.1])
    F.filtered_predictions(model, images, [("sal_pimienta", 3)], "mediana", root=tmp_path)
    original = cv2.imread(str(tmp_path / "i0.png"))
    raw = AC.to_cnn_input(audit.degrade(original, "sal_pimienta", 3, "i0.png"))
    assert not np.array_equal(model.batches[0][0], raw)


def test_predictions_have_one_row_per_image_and_condition_and_name_the_correction(tmp_path):
    images = _images(tmp_path)
    out = F.filtered_predictions(FixedModel([0.7, 0.1, 0.1, 0.1]), images, PLAN, "mediana", root=tmp_path)
    assert len(out) == len(images) * len(PLAN)
    assert set(out["correccion"]) == {"mediana"} and set(out["pred"]) == {"normal"}


def test_an_unknown_correction_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown correction"):
        F.filtered_predictions(FixedModel([1, 0, 0, 0]), _images(tmp_path, 1), PLAN, "inventada", root=tmp_path)


def _preds(pred_for_tapada, correction=None):
    rows = []
    for name, level in PLAN:
        for label, pred in (("normal", "normal"), ("tapada", pred_for_tapada), ("tapada", pred_for_tapada)):
            row = {"label": label, "perturbacion": name, "nivel": level, "pred": pred,
                   "p_normal": 0.9 if pred == "normal" else 0.1, "p_desenfocada": 0.0, "p_girada": 0.0,
                   "p_tapada": 0.0}
            if correction:
                row["correccion"] = correction
            rows.append(row)
    return pd.DataFrame(rows)


def test_before_after_puts_both_scores_side_by_side_and_their_difference():
    table = F.before_after(_preds("girada"), _preds("tapada", "mediana"))
    assert list(zip(table["perturbacion"], table["nivel"])) == PLAN
    assert (table["correccion"] == "mediana").all()
    np.testing.assert_allclose(table["accuracy_despues"], 1.0)
    np.testing.assert_allclose(table["accuracy_antes"], 1 / 3)
    np.testing.assert_allclose(table["delta_f1"], table["f1_despues"] - table["f1_antes"])
    assert (table["delta_f1"] > 0).all()


def test_before_after_ignores_unfiltered_conditions_that_were_not_corrected():
    unfiltered = pd.concat([_preds("girada"), _preds("girada").assign(perturbacion="gamma")])
    table = F.before_after(unfiltered, _preds("tapada", "mediana"))
    assert set(table["perturbacion"]) == {"sal_pimienta", "jpeg"}
