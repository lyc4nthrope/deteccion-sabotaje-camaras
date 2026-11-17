"""Tests for the Medellín normal-image selection (T17)."""

import pandas as pd
import pytest

from src import medellin as M


def _inventory(counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for group, n in counts.items():
        for i in range(n):
            rows.append({"archivo": f"imagen{group}_20230101_{i:06d}.jpg"})
    return pd.DataFrame(rows)


def test_allocation_takes_everything_when_total_is_large_enough():
    sizes = pd.Series({1: 3, 2: 5})
    assert M.allocation_per_group(sizes, 100).tolist() == [3, 5]


def test_allocation_caps_big_groups_and_keeps_small_ones_whole():
    sizes = pd.Series({1: 2, 2: 100, 3: 100})
    alloc = M.allocation_per_group(sizes, 60)
    assert alloc[1] == 2
    assert alloc.sum() == 60
    assert alloc[2] == alloc[3] == 29


def test_select_returns_exact_total_and_covers_every_group():
    inv = _inventory({1: 3, 2: 40, 3: 80, 4: 1})
    sel = M.select_normal(inv, total=50)
    assert len(sel) == 50
    assert set(sel["grupo"]) == {1, 2, 3, 4}


def test_select_has_no_duplicate_files():
    sel = M.select_normal(_inventory({1: 30, 2: 30}), total=40)
    assert not sel["archivo"].duplicated().any()


def test_select_is_deterministic():
    inv = _inventory({1: 30, 2: 30, 3: 7})
    assert M.select_normal(inv, 25).equals(M.select_normal(inv, 25))


def test_names_outside_the_pattern_are_rejected():
    with pytest.raises(ValueError):
        M.add_group_columns(pd.DataFrame({"archivo": ["foto.jpg"]}))


def test_selection_is_spread_over_time_inside_a_group():
    sel = M.select_normal(_inventory({1: 100}), total=10)
    instants = sel["instante"].tolist()
    assert instants[0] == min(instants) and instants[-1] == max(instants)


# ---------------------------------------------------------------- selection from the camera groups (T17b)

import numpy as np  # noqa: E402

from src import data  # noqa: E402


def _groups(spec: dict[int, tuple[int, bool]], step_minutes: int = 5) -> pd.DataFrame:
    """spec: cluster -> (images, reliable). Every 5th image of a cluster (from the first) is a reference."""
    rows = []
    for cluster, (n, reliable) in spec.items():
        for i in range(n):
            moment = pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=step_minutes * i)
            rows.append(
                {
                    "archivo": f"c{cluster}_{i:03d}.jpg",
                    "grupo_nombre": f"imagen{cluster}",
                    "cluster": cluster,
                    "instante": moment.strftime("%Y%m%d%H%M%S"),
                    "conservado": True,
                    "es_referencia": i % 20 == 0,
                    "confiable": reliable,
                }
            )
    return pd.DataFrame(rows)


def test_selection_from_groups_never_picks_a_reference_or_a_doubtful_group():
    groups = _groups({1: (100, True), 2: (100, False), 3: (100, True)})
    sel = M.select_from_groups(groups, total=60, min_gap_minutes=10)
    assert not set(sel["archivo"]) & set(groups.loc[groups["es_referencia"], "archivo"])
    assert set(sel["archivo"]) <= set(groups.loc[groups["confiable"], "archivo"])
    assert set(sel["grupo"]) == {1, 3}


def test_selection_from_groups_keeps_a_time_gap_to_every_reference_of_its_group():
    groups = _groups({1: (200, True)})
    sel = M.select_from_groups(groups, total=50, min_gap_minutes=10)
    refs = pd.to_datetime(groups.loc[groups["es_referencia"], "instante"], format="%Y%m%d%H%M%S")
    for moment in pd.to_datetime(sel["instante"], format="%Y%m%d%H%M%S"):
        assert (abs(refs - moment) >= pd.Timedelta(minutes=10)).all()


def test_selection_from_groups_uses_the_cluster_as_the_group_and_keeps_the_name_for_traceability():
    sel = M.select_from_groups(_groups({7: (80, True), 9: (80, True)}), total=40, min_gap_minutes=10)
    assert set(sel["grupo"]) == {7, 9}
    assert set(sel["grupo_nombre"]) == {"imagen7", "imagen9"}


def test_selection_from_groups_is_deterministic_and_respects_the_total():
    groups = _groups({1: (150, True), 2: (150, True)})
    first = M.select_from_groups(groups, total=100, min_gap_minutes=10)
    assert first.equals(M.select_from_groups(groups, total=100, min_gap_minutes=10))
    assert len(first) == 100 and not first["archivo"].duplicated().any()


def test_the_committed_selection_has_no_leak_with_the_references_of_the_groups():
    """Review of PR #31: 116 of the 600 images of the cross test were also references of their group."""
    selection = pd.read_csv(data.SPLITS_DIR / "medellin_normales.csv", dtype={"instante": str})
    groups = pd.read_csv(data.SPLITS_DIR / "medellin_grupos.csv", dtype={"instante": str})
    refs = groups[groups["es_referencia"]]
    assert not set(selection["archivo"]) & set(refs["archivo"])
    assert groups.set_index("archivo").loc[selection["archivo"], "confiable"].all()
    assert len(selection) == 600
    gap = pd.Timedelta(minutes=M.MIN_GAP_MINUTES)
    moments = pd.to_datetime(selection["instante"], format="%Y%m%d%H%M%S")
    ref_moments = pd.to_datetime(refs["instante"], format="%Y%m%d%H%M%S")
    by_cluster = {c: ref_moments[refs["cluster"] == c].to_numpy() for c in refs["cluster"].unique()}
    closest = [np.abs(by_cluster[c] - np.datetime64(m)).min() for c, m in zip(selection["grupo"], moments)]
    assert min(closest) >= np.timedelta64(int(gap.total_seconds()), "s")
