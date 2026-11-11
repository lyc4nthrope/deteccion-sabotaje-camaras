"""Tests for the before/after evaluation of the correction filters (T92)."""

import numpy as np
import pandas as pd
import pytest

from src import audit, corrections_audit as CA, perturbations as P


def _image(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (720, 1280, 3), dtype=np.uint8)


def test_every_perturbation_has_at_least_one_paired_correction():
    assert set(CA.PAIRED) == set(P.PERTURBATIONS)
    assert all(len(names) >= 1 for names in CA.PAIRED.values())


def test_pairs_follow_the_theory():
    assert "mediana" in CA.PAIRED["sal_pimienta"]
    assert {"mediana", "bilateral"} <= set(CA.PAIRED["ruido_gaussiano"])
    assert "clahe" in CA.PAIRED["brillo_oscuro"]


def test_gamma_is_paired_with_the_inverse_of_its_own_level():
    assert CA.corrections_for("gamma", 2, available={"gamma_inverso_nivel_2", "clahe"}) == ["gamma_inverso_nivel_2", "clahe"]


def test_missing_corrections_are_skipped():
    # brillo_claro_inverso_* arrives with PR #18; until then it must be skipped, not crash.
    assert CA.corrections_for("brillo_claro", 3, available={"clahe"}) == []


def test_runs_include_every_correction_on_the_clean_images():
    runs = CA.runs(available=set(CA.ALL_CORRECTIONS))
    clean = {c for p, l, c in runs if (p, l) == audit.ORIGINAL and c != CA.NONE}
    assert clean == set(CA.CLEAN_CHECK)


def test_runs_include_the_uncorrected_baseline_of_each_condition():
    runs = CA.runs(available=set(CA.ALL_CORRECTIONS))
    baselines = {(p, l) for p, l, c in runs if c == CA.NONE}
    assert baselines == set(audit.conditions())


def test_correct_applies_the_filter_after_the_perturbation():
    image = _image()
    degraded = audit.degrade(image, "sal_pimienta", 3, "a.jpg")
    corrected = CA.correct(image, "sal_pimienta", 3, "mediana", "a.jpg")
    # The median removes most of the 0 / 255 impulses that salt-and-pepper adds.
    impulses = lambda x: np.mean((x == 0) | (x == 255))
    assert impulses(corrected) < impulses(degraded) / 3


def test_no_correction_returns_the_degraded_image():
    image = _image()
    assert np.array_equal(CA.correct(image, "jpeg", 2, CA.NONE, "a.jpg"), audit.degrade(image, "jpeg", 2, "a.jpg"))


def test_before_after_table_pairs_each_correction_with_its_baseline():
    rows = []
    for correction, f1 in ((CA.NONE, 0.3), ("mediana", 0.9)):
        rows.append({"perturbacion": "sal_pimienta", "nivel": 3, "correccion": correction, "f1_macro": f1,
                     "fpr_normal": 0.2 if correction == CA.NONE else 0.01, "accuracy": f1, "auc_binaria": 0.9, "n": 10})
    table = CA.before_after(pd.DataFrame(rows))
    row = table.iloc[0]
    assert row["correccion"] == "mediana"
    assert row["f1_antes"] == pytest.approx(0.3) and row["f1_despues"] == pytest.approx(0.9)
    assert row["mejora_f1"] == pytest.approx(0.6)
    assert row["fpr_antes"] == pytest.approx(0.2) and row["fpr_despues"] == pytest.approx(0.01)


def test_saturation_is_total_and_unrecoverable_when_the_attack_clips_everything():
    image = np.full((720, 1280, 3), 200, np.uint8)  # x 2.0 -> 400, clipped to 255 everywhere
    out = CA.saturation(image, 3)
    assert out["saturados"] == 1.0
    assert out["saturados_original"] == 0.0
    assert out["error_en_saturados"] == pytest.approx(72.0)  # 255 / 2 rounds to 128, the original was 200


def test_saturation_is_zero_and_the_inverse_is_exact_when_nothing_clips():
    image = np.full((720, 1280, 3), 100, np.uint8)  # x 1.3 -> 130, no clipping
    out = CA.saturation(image, 1)
    assert out["saturados"] == 0.0
    assert out["error_medio"] == 0.0
    assert out["error_en_saturados"] == 0.0
