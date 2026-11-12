"""Tests for the CNN robustness audit on the degraded test set (T71)."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src import audit, audit_cnn as AC, cnn


class FixedModel:
    """Stand-in for the Keras model: always the same probabilities, and it records what it was given."""

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=np.float32)
        self.batches = []

    def predict(self, batch, verbose=0):
        self.batches.append(np.asarray(batch))
        return np.tile(self.proba, (len(batch), 1))


def _write_images(tmp_path, n=2):
    rows = []
    for i in range(n):
        image = np.random.default_rng(i).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        rel = f"img_{i}.png"
        cv2.imwrite(str(tmp_path / rel), image)
        rows.append({"path": rel, "label": "normal" if i == 0 else "tapada"})
    return pd.DataFrame(rows)


def test_to_cnn_input_matches_cnn_load_images(tmp_path):
    image = np.random.default_rng(0).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), image)
    expected = cnn.load_images(["a.png"], root=tmp_path)[0]
    out = AC.to_cnn_input(image)
    assert out.shape == (cnn.IMAGE_SIZE, cnn.IMAGE_SIZE, 3)
    assert out.min() >= -1.0 and out.max() <= 1.0
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_predictions_cover_every_image_and_condition(tmp_path):
    images = _write_images(tmp_path)
    model = FixedModel([0.7, 0.1, 0.1, 0.1])
    out = AC.degraded_predictions(model, images, root=tmp_path)
    assert len(out) == len(images) * len(audit.conditions())
    assert set(out["pred"]) == {"normal"}
    assert len(model.batches) == len(images)  # one batch per image: its conditions together, low memory


def test_cnn_sees_exactly_the_degraded_frame_of_the_classic_audit(tmp_path):
    images = _write_images(tmp_path, n=1)
    model = FixedModel([0.25, 0.25, 0.25, 0.25])
    AC.degraded_predictions(model, images, root=tmp_path)
    original = cv2.imread(str(tmp_path / images["path"][0]))
    for i, (name, level) in enumerate(audit.conditions()):
        expected = AC.to_cnn_input(audit.degrade(original, name, level, images["path"][0]))
        np.testing.assert_array_equal(model.batches[0][i], expected)


def test_condition_table_uses_one_minus_p_normal_as_score_and_counts_predictions():
    rows = []
    for name, level in audit.conditions():
        rows.append({"path": "a", "label": "normal", "perturbacion": name, "nivel": level, "pred": "normal",
                     "p_normal": 0.9, "p_desenfocada": 0.05, "p_girada": 0.03, "p_tapada": 0.02})
        rows.append({"path": "b", "label": "tapada", "perturbacion": name, "nivel": level, "pred": "tapada",
                     "p_normal": 0.1, "p_desenfocada": 0.0, "p_girada": 0.0, "p_tapada": 0.9})
    table = AC.condition_table(pd.DataFrame(rows))
    assert list(zip(table["perturbacion"], table["nivel"])) == audit.conditions()
    assert (table["modelo"] == "cnn").all()
    assert (table["accuracy"] == 1.0).all()
    assert (table["auc_binaria"] == 1.0).all()  # the sabotaged image always scores higher than the normal one
    assert (table["pred_normal"] == 1).all() and (table["pred_tapada"] == 1).all()


def test_merge_keeps_classic_rows_and_replaces_old_cnn_rows():
    columns = ["modelo", "perturbacion", "nivel", "accuracy"]
    existing = pd.DataFrame([["clasico", "original", 0, 0.99], ["cnn", "original", 0, 0.10]], columns=columns)
    new = pd.DataFrame([["cnn", "original", 0, 0.98]], columns=columns)
    merged = AC.merge_into_table(existing, new)
    assert merged.loc[merged["modelo"] == "clasico", "accuracy"].tolist() == [0.99]
    assert merged.loc[merged["modelo"] == "cnn", "accuracy"].tolist() == [0.98]
    assert len(merged) == 2


def test_condition_table_rejects_nothing_but_needs_known_labels():
    bad = pd.DataFrame([{"path": "a", "label": "otra", "perturbacion": "original", "nivel": 0, "pred": "normal",
                         "p_normal": 0.5, "p_desenfocada": 0.5, "p_girada": 0.0, "p_tapada": 0.0}])
    with pytest.raises(ValueError):
        AC.condition_table(bad)
