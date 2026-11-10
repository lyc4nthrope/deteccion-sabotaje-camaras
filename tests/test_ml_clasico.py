"""Tests for the supervised classifiers on the classic descriptors (T45, class 20)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import metrics, ml_clasico

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTORS = ROOT / "results" / "clasico" / "descriptores_ubicacion1.csv"
SPLIT = ROOT / "data" / "splits" / "split.csv"


def _frame(n_per_class=24, seed=0):
    """Separable synthetic descriptors with 4 blocks of consecutive images per class."""
    rng = np.random.default_rng(seed)
    centers = {
        "normal": (400, 30, 6.8, 0.2, 40, 0.05, 200, 190, 0.1),
        "desenfocada": (5, 16, 5.9, 0.08, 18, 0.03, 1, 0, np.nan),
        "girada": (600, 40, 7.2, 0.23, 70, 0.08, 8, 2, np.nan),
        "tapada": (4, 8, 4.9, 0.08, 1.5, 0.0, 0, 0, np.nan),
    }
    rows = []
    for label, center in centers.items():
        for i in range(n_per_class):
            values = [c * rng.uniform(0.9, 1.1) if not np.isnan(c) else c for c in center]
            rows.append(dict(zip(ml_clasico.RAW_COLUMNS, values), label=label, block=f"{label}_{i // 6:04d}"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- features


def test_features_log_scale_the_heavy_tailed_descriptors():
    frame = _frame()
    X = ml_clasico.prepare_features(frame)
    assert X["lap_var"].iloc[0] == pytest.approx(np.log1p(frame["lap_var"].iloc[0]))
    assert X["orb_matches"].iloc[0] == pytest.approx(np.log1p(frame["orb_matches"].iloc[0]))


def test_features_encode_a_missing_angle_as_a_flag():
    frame = _frame()
    X = ml_clasico.prepare_features(frame)
    missing = frame["angle"].isna()
    assert X.loc[missing, "angle_missing"].eq(1).all() and X.loc[~missing, "angle_missing"].eq(0).all()
    assert X.loc[missing, "abs_angle"].eq(0).all()
    assert X.loc[~missing, "abs_angle"].equals(frame.loc[~missing, "angle"].abs())
    assert not X.isna().any().any()


def test_features_have_a_fixed_column_order():
    assert list(ml_clasico.prepare_features(_frame()).columns) == ml_clasico.FEATURE_COLUMNS


# ---------------------------------------------------------------- grouped cross-validation


def test_cross_validation_never_splits_a_block():
    frame = _frame()
    for train_idx, valid_idx in ml_clasico.grouped_folds(frame):
        assert not set(frame["block"].iloc[train_idx]) & set(frame["block"].iloc[valid_idx])


def test_cross_validation_uses_five_folds():
    assert len(list(ml_clasico.grouped_folds(_frame()))) == ml_clasico.N_FOLDS == 5


# ---------------------------------------------------------------- models


def test_the_three_class_20_models_are_available():
    assert set(ml_clasico.MODELS) == {"knn", "svm", "naive_bayes"}


@pytest.mark.parametrize("name", ["knn", "svm", "naive_bayes"])
def test_each_model_learns_separable_descriptors(name):
    model = ml_clasico.fit(name, _frame(seed=0))
    test = _frame(seed=1)
    predicted = model.predict(ml_clasico.prepare_features(test))
    assert metrics.f1_macro(test["label"], predicted) > 0.95


def test_distance_models_standardize_the_features():
    model = ml_clasico.fit("knn", _frame())
    scaler = model.best_estimator_.named_steps["scale"]
    assert np.allclose(scaler.transform(ml_clasico.prepare_features(_frame())).mean(axis=0), 0, atol=1e-6)


def test_hyperparameters_are_chosen_by_grouped_cross_validation():
    model = ml_clasico.fit("svm", _frame())
    assert model.n_splits_ == ml_clasico.N_FOLDS
    assert model.scoring == "f1_macro"


def test_fit_rejects_rows_outside_train():
    frame = _frame().assign(set="val")
    with pytest.raises(ValueError, match="train"):
        ml_clasico.fit("knn", frame)


# ---------------------------------------------------------------- real descriptors


@pytest.mark.skipif(not (DESCRIPTORS.exists() and SPLIT.exists()), reason="Run python -m src.build_features first")
def test_real_validation_performance_is_reasonable():
    train, val = ml_clasico.load_sets()[:2]
    model = ml_clasico.fit("svm", train)
    predicted = model.predict(ml_clasico.prepare_features(val))
    assert metrics.f1_macro(val["label"], predicted) > 0.8
