"""Tests for the audit figures and cross confusion matrices (T72, T73)."""

import numpy as np
import pandas as pd
import pytest

from src import audit, audit_figures as AF, classic, metrics

T = classic.Thresholds(match_min=12.5, blur_max=18.9, cover_feature="hue_std", cover_threshold=5.4,
                       cover_direction="<", angle_max=0.94)


def _descriptors():
    """One normal and one covered image under every condition, with descriptors that the tree reads clearly."""
    rows = []
    for name, level in audit.conditions():
        rows.append({"path": "n.jpg", "label": "normal", "perturbacion": name, "nivel": level, "orb_matches": 200,
                     "angle": 0.1, "lap_var": 100.0, "hue_std": 20.0})
        rows.append({"path": "t.jpg", "label": "tapada", "perturbacion": name, "nivel": level, "orb_matches": 0,
                     "angle": np.nan, "lap_var": 2.0, "hue_std": 1.0})
    return pd.DataFrame(rows)


def _cnn():
    d = _descriptors()[["path", "label", "perturbacion", "nivel"]]
    normal = d["label"] == "normal"
    return d.assign(pred=np.where(normal, "normal", "tapada"), p_normal=np.where(normal, 0.9, 0.2))


def test_predictions_have_both_models_with_the_classic_rules_and_one_minus_p_normal():
    frame = AF.predictions(_descriptors(), T, _cnn())
    assert set(frame["modelo"]) == {"clasico", "cnn"}
    assert len(frame) == 2 * len(_descriptors())
    clasico = frame[frame["modelo"] == "clasico"]
    assert (clasico["pred"] == clasico["label"]).all()
    cnn = frame[frame["modelo"] == "cnn"]
    assert cnn.loc[cnn["label"] == "normal", "score"].round(6).eq(0.1).all()
    assert cnn.loc[cnn["label"] == "tapada", "score"].round(6).eq(0.8).all()


def test_cross_matrices_count_every_image_once_per_model_and_condition():
    frame = AF.predictions(_descriptors(), T, _cnn())
    table = AF.cross_matrices(frame)
    per_cell = table.groupby(["modelo", "perturbacion", "nivel"])["n"].sum()
    assert len(per_cell) == 2 * len(AF.CRITICAL)
    assert (per_cell == 2).all()
    hit = table[(table["real"] == "tapada") & (table["predicha"] == "tapada")]
    assert (hit["n"] == 1).all()
    assert len(table) == 2 * len(AF.CRITICAL) * len(metrics.LABELS) ** 2


def test_critical_conditions_are_the_original_and_level_3_of_each_perturbation():
    assert AF.CRITICAL[0] == audit.ORIGINAL
    assert {level for _, level in AF.CRITICAL[1:]} == {3}
    assert len(AF.CRITICAL) == 1 + len(audit.P.PERTURBATIONS)


def test_f1_by_level_starts_every_curve_at_the_original():
    rows = [{"modelo": m, "perturbacion": "original", "nivel": 0, "f1_macro": 0.9} for m in AF.MODELS]
    rows += [{"modelo": m, "perturbacion": p, "nivel": l, "f1_macro": 0.5}
             for m in AF.MODELS for p in audit.P.PERTURBATIONS for l in audit.P.LEVELS]
    levels = AF.f1_by_level(pd.DataFrame(rows))
    assert len(levels) == len(AF.MODELS) * len(audit.P.PERTURBATIONS) * 4
    assert (levels.loc[levels["nivel"] == 0, "f1_macro"] == 0.9).all()
    assert (levels.loc[levels["nivel"] > 0, "f1_macro"] == 0.5).all()


def test_figures_are_built_without_error():
    frame = AF.predictions(_descriptors(), T, _cnn())
    assert AF.plot_matrices(AF.cross_matrices(frame)) is not None
    assert AF.plot_roc(frame) is not None
