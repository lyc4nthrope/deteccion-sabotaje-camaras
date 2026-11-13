"""Tests for recovering the camera of each Medellín image by clustering (T17b)."""

import numpy as np
import pandas as pd
import pytest

from src import medellin_groups as G


def _scene(seed: int) -> np.ndarray:
    """A fixed 'camera view': coarse random blocks, upscaled so it has structure at thumbnail size."""
    small = np.random.default_rng(seed).integers(0, 256, (9, 16, 3), dtype=np.uint8)
    return np.kron(small, np.ones((40, 40, 1), dtype=np.uint8))  # 360 x 640


def _shot(scene: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same view at another time of day: different gain and offset plus sensor noise."""
    gain, offset = rng.uniform(0.5, 1.2), rng.uniform(-20, 20)
    noisy = scene.astype(float) * gain + offset + rng.normal(0, 4, scene.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- descriptor


def test_descriptor_is_a_standardized_gray_thumbnail():
    d = G.thumbnail_descriptor(_scene(0))
    assert d.shape == (32 * 18,) and d.dtype == np.float32
    assert abs(float(d.mean())) < 1e-4 and float(d.std()) == pytest.approx(1.0, abs=1e-3)


def test_descriptor_ignores_global_brightness_changes():
    scene = _scene(1)
    brighter = np.clip(scene.astype(float) * 0.6 + 15, 0, 255).astype(np.uint8)
    a, b = G.thumbnail_descriptor(scene), G.thumbnail_descriptor(brighter)
    assert float(np.corrcoef(a, b)[0, 1]) > 0.99


def test_descriptor_of_a_flat_image_is_zeros_not_nan():
    d = G.thumbnail_descriptor(np.full((360, 640, 3), 120, np.uint8))
    assert np.all(d == 0) and np.isfinite(d).all()


# ---------------------------------------------------------------- clustering


def _three_cameras():
    rng = np.random.default_rng(0)
    scenes = [_scene(s) for s in (10, 20, 30)]
    truth = np.repeat([0, 1, 2], 15)
    X = np.stack([G.thumbnail_descriptor(_shot(scenes[c], rng)) for c in truth])
    return X, truth


def test_ward_recovers_the_cameras_when_k_is_right():
    X, truth = _three_cameras()
    labels = G.cut_tree(G.ward_tree(X), 3)
    assert G.adjusted_rand(truth, labels) == pytest.approx(1.0)
    assert set(labels) == {1, 2, 3}


def test_evaluate_k_scores_each_candidate_number_of_groups():
    X, truth = _three_cameras()
    table = G.evaluate_k(G.ward_tree(X), X, [2, 3, 6], reference=truth)
    assert list(table["k"]) == [2, 3, 6]
    assert {"silhouette", "ari_vs_nombre"} <= set(table.columns)
    best = table.sort_values("silhouette", ascending=False).iloc[0]
    assert int(best["k"]) == 3 and best["ari_vs_nombre"] == pytest.approx(1.0)


# ---------------------------------------------------------------- groups and references


def _inventory(counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for cluster, n in counts.items():
        for i in range(n):
            rows.append({"archivo": f"c{cluster}_{i:02d}.jpg", "cluster": cluster, "instante": f"2022100{i % 9 + 1}{i:06d}"})
    return pd.DataFrame(rows)


def test_groups_below_the_minimum_are_discarded_and_get_no_references():
    out = G.assign_groups(_inventory({1: 9, 2: 12}), min_size=10, n_refs=8)
    small, big = out[out["cluster"] == 1], out[out["cluster"] == 2]
    assert not small["conservado"].any() and not small["es_referencia"].any()
    assert big["conservado"].all() and int(big["es_referencia"].sum()) == 8


def test_references_are_spread_over_the_time_order_of_the_group():
    out = G.assign_groups(_inventory({1: 40}), min_size=10, n_refs=8)
    ordered = out.sort_values("instante").reset_index(drop=True)
    refs = ordered.index[ordered["es_referencia"]].tolist()
    assert refs[0] == 0 and refs[-1] == 39 and len(refs) == 8
    assert all(b - a >= 4 for a, b in zip(refs, refs[1:]))


def test_every_kept_group_keeps_images_left_to_evaluate():
    out = G.assign_groups(_inventory({1: 10, 2: 30}), min_size=10, n_refs=8)
    kept = out[out["conservado"]]
    assert (kept.groupby("cluster")["es_referencia"].apply(lambda s: (~s).sum()) >= 2).all()


# ---------------------------------------------------------------- do the references match their group?


class _FakeMatcher:
    """Stands in for ReferenceMatcher: 30 matches when the image has the same marker as the references, else 2."""

    def __init__(self, references_gray):
        self.marker = int(references_gray[0][0, 0])

    def match(self, gray):
        return {"orb_matches": 30 if int(gray[0, 0]) == self.marker else 2}


def _loader(markers: dict[str, int]):
    return lambda name: np.full((360, 640), markers[name], np.uint8)


def test_match_rate_is_one_when_every_image_matches_the_references_of_its_group():
    groups = G.assign_groups(_inventory({1: 20}), min_size=10, n_refs=8)
    rates = G.match_rate_per_group(groups, _loader({n: 7 for n in groups["archivo"]}), _FakeMatcher, match_min=12.5)
    assert rates.loc[1, "tasa"] == pytest.approx(1.0) and rates.loc[1, "evaluadas"] == 12


def test_match_rate_drops_when_a_group_mixes_two_scenes():
    groups = G.assign_groups(_inventory({1: 20}), min_size=10, n_refs=8)
    ordered = groups.sort_values("instante")
    foreign = ordered[~ordered["es_referencia"]].head(6)["archivo"]
    markers = {n: (99 if n in set(foreign) else 7) for n in groups["archivo"]}
    rates = G.match_rate_per_group(groups, _loader(markers), _FakeMatcher, match_min=12.5)
    assert rates.loc[1, "tasa"] == pytest.approx(0.5)


def test_match_rate_samples_at_most_the_requested_images_and_skips_discarded_groups():
    groups = G.assign_groups(_inventory({1: 50, 2: 5}), min_size=10, n_refs=8)
    rates = G.match_rate_per_group(groups, _loader({n: 7 for n in groups["archivo"]}), _FakeMatcher, sample=10)
    assert list(rates.index) == [1] and rates.loc[1, "evaluadas"] == 10
