"""Tests for the Grad-CAM explanations of the CNN (T67). The model is untrained (no network, no weights)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from src import cnn  # noqa: E402
from src import explain_cnn as X  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return cnn.build_model(weights=None)


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, (4, cnn.IMAGE_SIZE, cnn.IMAGE_SIZE, 3)).astype(np.float32)


# ---------------------------------------------------------------- Grad-CAM


def test_heatmaps_are_one_per_image_on_the_7x7_feature_grid_scaled_to_0_1(model, batch):
    out = X.grad_cam(model, batch)
    assert out["heatmaps"].shape == (4, 7, 7)
    assert np.isfinite(out["heatmaps"]).all()
    assert out["heatmaps"].min() >= 0.0 and out["heatmaps"].max() <= 1.0 + 1e-6


def test_prediction_and_probabilities_match_the_model(model, batch):
    out = X.grad_cam(model, batch)
    proba = model.predict(batch, verbose=0)
    np.testing.assert_allclose(out["proba"], proba, atol=1e-4)
    # An untrained model gives near-equal probabilities, so compare with a tolerance instead of the exact argmax.
    np.testing.assert_allclose(proba[np.arange(len(batch)), out["pred"]], proba.max(axis=1), atol=1e-4)


def test_gradcam_equals_the_class_activation_map_of_a_global_pooling_head(model, batch):
    """With global average pooling + a dense layer, Grad-CAM reduces to relu(sum_k W[k, c] * A_k)."""
    import tensorflow as tf

    base = X.base_of(model)
    conv = tf.keras.Model(base.input, base.get_layer("out_relu").output)(batch).numpy()  # (n, 7, 7, 1280)
    kernel = model.layers[-1].get_weights()[0]  # (1280, 4)
    out = X.grad_cam(model, batch)
    for i in range(len(batch)):
        cam = np.maximum(conv[i] @ kernel[:, out["pred"][i]], 0.0)
        expected = cam / cam.max() if cam.max() > 0 else cam
        np.testing.assert_allclose(out["heatmaps"][i], expected, atol=1e-3)


def test_a_chosen_class_gets_the_map_of_that_class_not_the_predicted_one(model, batch):
    """The map must be the CAM of the requested class (kernel column), not just an echo of the requested index."""
    import tensorflow as tf

    base = X.base_of(model)
    conv = tf.keras.Model(base.input, base.get_layer("out_relu").output)(batch).numpy()
    kernel = model.layers[-1].get_weights()[0]
    predicted = X.grad_cam(model, batch)
    other = [(int(p) + 1) % 4 for p in predicted["pred"]]
    out = X.grad_cam(model, batch, class_index=other)
    assert list(out["explained"]) == other
    assert not np.allclose(out["heatmaps"], predicted["heatmaps"], atol=1e-3)
    for i, c in enumerate(other):
        cam = np.maximum(conv[i] @ kernel[:, c], 0.0)
        expected = cam / cam.max() if cam.max() > 0 else cam
        np.testing.assert_allclose(out["heatmaps"][i], expected, atol=1e-3)


# ---------------------------------------------------------------- overlay


def test_overlay_keeps_the_image_size_and_dtype():
    image = np.random.default_rng(1).integers(0, 256, (135, 240, 3), dtype=np.uint8)
    out = X.overlay(image, np.random.default_rng(2).random((7, 7)))
    assert out.shape == image.shape and out.dtype == np.uint8


def test_an_all_zero_heatmap_leaves_the_image_untouched():
    image = np.random.default_rng(3).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    np.testing.assert_array_equal(X.overlay(image, np.zeros((7, 7))), image)


# ---------------------------------------------------------------- where the model looks


def test_attention_stats_locate_a_top_left_blob_in_the_upper_third():
    heat = np.zeros((1, 7, 7))
    heat[0, 0:2, 0:2] = 1.0
    stats = X.attention_stats(heat)
    assert stats.loc[0, "centro_y"] < 0.25 and stats.loc[0, "centro_x"] < 0.25
    assert stats.loc[0, "tercio_superior"] == pytest.approx(1.0)
    assert stats.loc[0, "tercio_superior"] + stats.loc[0, "tercio_medio"] + stats.loc[0, "tercio_inferior"] == pytest.approx(1.0)


def test_a_uniform_heatmap_puts_exactly_one_third_in_each_third_of_the_image():
    """7 rows do not split into 3 whole thirds: the middle row of each boundary is shared proportionally."""
    stats = X.attention_stats(np.ones((1, 7, 7)))
    for column in ("tercio_superior", "tercio_medio", "tercio_inferior"):
        assert stats.loc[0, column] == pytest.approx(1 / 3)


def test_attention_stats_of_an_empty_heatmap_do_not_divide_by_zero():
    stats = X.attention_stats(np.zeros((1, 7, 7)))
    assert stats.loc[0, "masa"] == 0.0 and np.isfinite(stats.loc[0].to_numpy(dtype=float)).all()


def test_a_uniform_heatmap_is_less_concentrated_than_a_single_cell():
    spread = X.attention_stats(np.ones((1, 7, 7))).loc[0, "concentracion"]
    point = np.zeros((1, 7, 7))
    point[0, 3, 3] = 1.0
    assert X.attention_stats(point).loc[0, "concentracion"] > spread


# ---------------------------------------------------------------- choosing examples


def _table():
    rows = []
    for label, n in (("normal", 10), ("desenfocada", 6), ("girada", 2)):
        rows += [{"path": f"{label}_{i}.jpg", "label": label} for i in range(n)]
    return pd.DataFrame(rows)


def test_pick_examples_takes_n_per_class_deterministically_and_all_when_there_are_fewer():
    first = X.pick_examples(_table(), per_class=4, seed=0)
    second = X.pick_examples(_table(), per_class=4, seed=0)
    assert first["path"].tolist() == second["path"].tolist()
    assert first.groupby("label").size().to_dict() == {"desenfocada": 4, "girada": 2, "normal": 4}
