"""Tests for the CNN data and model (T60, T61)."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src import cnn


def test_read_split_keeps_only_train_val_and_test(tmp_path):
    split = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "label": ["normal", "tapada", "girada", "normal"],
            "set": ["train", "val", "descartada", "test"],
        }
    ).to_csv(split, index=False)
    table = cnn.read_split(split)
    assert set(table["set"]) == {"train", "val", "test"}
    assert len(table) == 3
    assert table.loc[table["label"] == "tapada", "y"].iloc[0] == cnn.CLASSES.index("tapada")


def test_read_split_rejects_unknown_labels(tmp_path):
    split = tmp_path / "split.csv"
    pd.DataFrame({"path": ["a.jpg"], "label": ["rota"], "set": ["train"]}).to_csv(split, index=False)
    with pytest.raises(ValueError):
        cnn.read_split(split)


def test_class_weights_balance_the_classes():
    labels = np.array([0] * 90 + [2] * 10)
    weights = cnn.class_weights(labels)
    assert weights[2] > weights[0]
    assert weights[0] * 90 == pytest.approx(weights[2] * 10)


def test_load_images_gives_224_rgb_in_minus_one_to_one(tmp_path):
    name = "imagen.png"  # PNG: JPEG would shift the pure red by one level
    frame = np.zeros((120, 200, 3), np.uint8)
    frame[..., 2] = 255  # red in BGR
    cv2.imwrite(str(tmp_path / name), frame)
    batch = cnn.load_images([name], root=tmp_path)
    assert batch.shape == (1, cnn.IMAGE_SIZE, cnn.IMAGE_SIZE, 3)
    assert batch.min() >= -1.0 and batch.max() <= 1.0
    assert batch[0, 0, 0, 0] == pytest.approx(1.0)  # red channel first after BGR -> RGB
    assert batch[0, 0, 0, 2] == pytest.approx(-1.0)


def test_load_images_fails_clearly_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        cnn.load_images(["nope.jpg"], root=tmp_path)


def test_model_has_four_outputs_and_frozen_base_by_default():
    model = cnn.build_model(weights=None)
    assert model.output_shape == (None, len(cnn.CLASSES))
    assert model.layers[0].trainable is False
    assert model.layers[0].count_params() > 1_000_000  # MobileNetV2 base


def test_model_unfreezes_the_last_layers_on_request():
    model = cnn.build_model(trainable_layers=5, weights=None)
    base = model.layers[0]
    assert base.trainable is True
    assert any(layer.trainable for layer in base.layers[-5:])
    assert not any(layer.trainable for layer in base.layers[:-5])


def test_parameter_counts_match_the_architecture_section_of_docs_cnn():
    """docs/cnn.md (T65) quotes these numbers for the defense: if the model changes, the document must change."""
    import numpy as np

    model = cnn.build_model(weights=None)
    count = lambda weights: sum(int(np.prod(w.shape)) for w in weights)
    assert model.count_params() == 2_263_108
    assert count(model.trainable_weights) == 5_124 == 1280 * len(cnn.CLASSES) + len(cnn.CLASSES)
    assert count(model.non_trainable_weights) == 2_257_984


def test_base_downsamples_by_32_with_five_stride_2_layers_and_ten_residual_blocks():
    import tensorflow as tf

    base = cnn.build_model(weights=None).layers[0]
    strided = [
        layer.output.shape[1]
        for layer in base.layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
        and tuple(layer.get_config()["strides"]) == (2, 2)
    ]
    assert strided == [112, 56, 28, 14, 7]  # 224 / 2**5 = 7
    assert tuple(base.get_layer("out_relu").output.shape[1:]) == (7, 7, 1280)
    assert sum(isinstance(layer, tf.keras.layers.Add) for layer in base.layers) == 10
