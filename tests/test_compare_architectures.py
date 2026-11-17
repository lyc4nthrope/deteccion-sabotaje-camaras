"""Tests for the architecture comparison of the docs (T65). The models are fakes: no TensorFlow needed."""

import numpy as np
import pandas as pd
import pytest

from src import compare_architectures as CA


def test_time_model_discards_the_warm_up_and_returns_the_median_in_milliseconds():
    clock = iter([0.0, 0.010, 0.0, 0.030, 0.0, 0.020])  # three timed runs: 10, 30 and 20 ms
    calls = []
    median = CA.time_model(lambda x: calls.append(1), np.zeros((1, 2)), warmup=2, runs=3, timer=lambda: next(clock))
    assert len(calls) == 2 + 3  # warm-up calls are made but not timed
    assert median == pytest.approx(20.0)


def test_comparison_table_has_the_ratio_against_the_first_model():
    table = CA.comparison_table([("A", 100, 10.0), ("B", 400, 25.0)], repetitions=30)
    assert list(table.columns) == ["modelo", "parametros", "ms_mediana", "veces_el_primero", "repeticiones"]
    assert table["veces_el_primero"].tolist() == [1.0, 2.5]
    assert table["repeticiones"].tolist() == [30, 30]


def test_comparison_table_rejects_an_empty_list():
    with pytest.raises(ValueError):
        CA.comparison_table([], repetitions=30)
