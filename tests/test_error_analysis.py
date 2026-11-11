"""Tests for the classic-module error analysis (T44)."""

import numpy as np
import pandas as pd

from src import classic
from src import error_analysis as E

T = classic.Thresholds(
    match_min=12.5, blur_max=18.0, cover_feature="hue_std", cover_threshold=5.0, cover_direction="<", angle_max=0.94
)


def _image(**values) -> pd.Series:
    base = {"orb_matches": 30.0, "angle": 0.1, "lap_var": 50.0, "hue_std": 9.0, "label": "normal", "set": "test"}
    return pd.Series({**base, **values})


def test_summary_counts_errors_by_pair_and_rule():
    errors = pd.DataFrame(
        {
            "label": ["girada", "girada", "normal"],
            "pred": ["normal", "normal", "girada"],
            "regla": ["coincide_y_giro_dentro_del_limite"] * 2 + ["coincide_pero_giro_grande"],
            "descriptor_decisivo": ["orb_matches", "orb_matches", "angle"],
        }
    )
    out = E.summary(errors)
    girada = out[(out["label"] == "girada") & (out["descriptor_decisivo"] == "orb_matches")]
    assert int(girada["n"].iloc[0]) == 2
    assert int(out["n"].sum()) == 3


def test_summary_of_no_errors_is_empty():
    empty = pd.DataFrame(columns=["label", "pred", "regla", "descriptor_decisivo"])
    assert E.summary(empty).empty


def test_girada_predicted_normal_is_explained_by_matches_not_by_skin_ratio():
    # Small rotation (0.37 < 0.94) and enough matches: the tree decided normal by the matches rule.
    rule = E.decisive_rule(_image(label="girada", angle=0.37, orb_matches=30.0), T)
    assert rule["regla"] == "coincide_y_giro_dentro_del_limite"
    assert rule["descriptor_decisivo"] == "orb_matches"
    assert rule["resultado"] == "normal"
    assert rule["margen"] == 17.5


def test_large_rotation_is_explained_by_angle():
    rule = E.decisive_rule(_image(angle=2.0), T)
    assert rule["regla"] == "coincide_pero_giro_grande"
    assert rule["descriptor_decisivo"] == "angle"
    assert rule["margen"] == 1.06


def test_matches_without_reliable_angle_are_explained_by_angle():
    rule = E.decisive_rule(_image(angle=np.nan), T)
    assert rule["regla"] == "coincide_sin_angulo_fiable"
    assert rule["resultado"] == "girada"


def test_no_matches_and_sharp_is_explained_by_sharpness():
    rule = E.decisive_rule(_image(orb_matches=3.0, lap_var=60.0), T)
    assert rule["regla"] == "sin_coincidencias_y_nitida"
    assert rule["descriptor_decisivo"] == "lap_var"
    assert rule["resultado"] == "girada"


def test_no_matches_and_blurry_is_explained_by_cover_rule():
    tapada = E.decisive_rule(_image(orb_matches=3.0, lap_var=5.0, hue_std=2.0), T)
    desenfocada = E.decisive_rule(_image(orb_matches=3.0, lap_var=5.0, hue_std=9.0), T)
    assert tapada["resultado"] == "tapada" and desenfocada["resultado"] == "desenfocada"
    assert tapada["descriptor_decisivo"] == "hue_std"


def test_analyse_lists_only_errors_with_the_rule_that_decided_them():
    table = pd.DataFrame(
        [
            {"path": "a.jpg", "label": "girada", "set": "test", "orb_matches": 30.0, "angle": 0.37,
             "lap_var": 50.0, "hue_std": 9.0},
            {"path": "b.jpg", "label": "normal", "set": "test", "orb_matches": 30.0, "angle": 0.1,
             "lap_var": 50.0, "hue_std": 9.0},
        ]
    )
    errors = E.analyse(table, T, "test")
    assert list(errors["path"]) == ["a.jpg"]
    assert errors["descriptor_decisivo"].iloc[0] == "orb_matches"
