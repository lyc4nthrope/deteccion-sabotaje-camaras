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
