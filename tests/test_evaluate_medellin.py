"""Tests for the Medellín cross test (T76): case table and per-variant recall."""

import numpy as np
import pandas as pd
import pytest

from src import evaluate_medellin as E


def _normales():
    return pd.DataFrame({"archivo": ["a.jpg", "b.jpg"], "grupo": [1, 2]})


def _manifest():
    rows = []
    for name, group in (("a.jpg", 1), ("b.jpg", 2)):
        for label in E.VARIANTS:
            rows.append({"archivo": name, "grupo": group, "etiqueta": label, "parametros": "{}"})
    return pd.DataFrame(rows)


def test_cases_have_the_normal_image_and_each_variant_with_the_bremen_class():
    cases = E.build_cases(_normales(), _manifest())
    assert len(cases) == 2 * 5
    assert set(cases["label"]) == {"normal", "desenfocada", "girada", "tapada"}
    assert (cases[cases["variante"] == "tapada_parcial"]["label"] == "tapada").all()
    assert (cases[cases["variante"] == "normal"]["label"] == "normal").all()


def test_every_case_points_to_the_file_of_its_variant():
    cases = E.build_cases(_normales(), _manifest())
    normal = cases[(cases["variante"] == "normal") & (cases["archivo"] == "a.jpg")]["path"].iloc[0]
    girada = cases[(cases["variante"] == "girada") & (cases["archivo"] == "a.jpg")]["path"].iloc[0]
    assert str(normal).endswith("images/a.jpg") or str(normal).endswith("images\\a.jpg")
    assert "girada" in str(girada) and str(girada).endswith("a.jpg")


def test_a_variant_without_its_normal_image_fails_loudly():
    manifest = _manifest()
    manifest.loc[0, "archivo"] = "z.jpg"
    with pytest.raises(ValueError, match="not in the normal list"):
        E.build_cases(_normales(), manifest)


def test_recall_by_variant_counts_each_variant_against_its_class():
    cases = E.build_cases(_normales(), _manifest())
    cases["pred"] = np.where(cases["variante"] == "girada", "normal", cases["label"])
    table = E.recall_by_variant(cases)
    got = dict(zip(table["variante"], table["recall"]))
    assert got["girada"] == 0.0 and got["normal"] == 1.0 and got["tapada_total"] == 1.0
