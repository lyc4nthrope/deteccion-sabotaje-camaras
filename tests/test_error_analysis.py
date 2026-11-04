"""Tests for the classic-module error analysis (T44)."""

import pandas as pd

from src import error_analysis as E


def test_summary_counts_errors_by_pair_and_descriptor():
    errors = pd.DataFrame(
        {
            "label": ["girada", "girada", "normal"],
            "pred": ["normal", "normal", "girada"],
            "descriptor_principal": ["orb_inliers", "orb_inliers", "hue_std"],
        }
    )
    out = E.summary(errors)
    girada = out[(out["label"] == "girada") & (out["descriptor_principal"] == "orb_inliers")]
    assert int(girada["n"].iloc[0]) == 2
    assert int(out["n"].sum()) == 3


def test_summary_of_no_errors_is_empty():
    empty = pd.DataFrame(columns=["label", "pred", "descriptor_principal"])
    assert E.summary(empty).empty
