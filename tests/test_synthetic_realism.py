"""The synthetic sabotage must look to the classic module like the real one (T18, review of PR #2).

A generator that only looks plausible makes the Medellín cross test (T76) measure the generator, not the camera:
the old pan was a zoom that the classic module called "normal" 92 % of the time, and a blur of sigma 4 to 8 left 88 %
of the blurred images "normal". Here the descriptors of the synthetic variants of Bremen test images are measured with
the real references and thresholds, and compared with the ranges of the real images of the same class (10th to 90th
percentile, results/clasico/descriptores_*.csv). Skipped when the Bremen images are not downloaded.
"""

import numpy as np
import pandas as pd
import pytest

from src import classic, data, features
from src import synthetic_tampering as S

pytestmark = pytest.mark.skipif(not data.BREMEN_ROOT.exists(), reason="the Bremen images are not downloaded")

N_IMAGES = 20


@pytest.fixture(scope="module")
def measured():
    thresholds = classic.load_thresholds(data.RESULTS_DIR / "clasico" / "umbrales.json")
    refs = pd.read_csv(data.SPLITS_DIR / "referencias.csv")
    matcher = features.ReferenceMatcher(
        [features.preprocess(features.load_image(data.BREMEN_ROOT / p))["gray"] for p in refs["path"]]
    )
    split = data.load_split()
    normals = split[(split["set"] == "test") & (split["label"] == "normal")].sample(N_IMAGES, random_state=7)
    rows = []
    for path in normals["path"]:
        image = features.load_image(data.BREMEN_ROOT / path)
        rows.append({**features.extract(image, matcher), "variante": "normal"})
        for label, (variant, _) in S.sabotage_all(image, S.file_seed(0, path)).items():
            rows.append({**features.extract(variant, matcher), "variante": label})
    table = pd.DataFrame(rows)
    table["pred"] = classic.predict(table, thresholds)
    return table


def _share(table, variant, predicted):
    group = table[table["variante"] == variant]
    return float(group["pred"].isin(predicted).mean())


def test_the_control_normal_images_are_called_normal(measured):
    assert _share(measured, "normal", {"normal"}) >= 0.95


def test_synthetic_blur_is_called_blurred_and_its_descriptors_are_in_the_real_range(measured):
    group = measured[measured["variante"] == "desenfocada"]
    assert _share(measured, "desenfocada", {"desenfocada"}) >= 0.85
    assert group["orb_matches"].median() <= 11  # real: median 0 to 2, 90th percentile 4 to 11
    assert 3.0 <= group["lap_var"].median() <= 28.0  # real: 3.1 to 28 between 10th and 90th percentile


def test_synthetic_pan_is_called_rotated_and_its_descriptors_are_in_the_real_range(measured):
    group = measured[measured["variante"] == "girada"]
    assert _share(measured, "girada", {"normal"}) <= 0.25  # real: 2 % to 16 % are called normal
    assert 5 <= group["orb_matches"].median() <= 56  # real: 10th percentile 5 (loc. 1) to 90th 56 (loc. 1)
    assert 0.5 <= group["angle"].abs().median() <= 9.7  # real: 0.5 to 9.7 degrees
    assert group["lap_var"].median() >= 30  # a pan keeps a sharp image, unlike a blur or a cover


def test_synthetic_total_cover_is_called_covered_with_the_sharpness_of_real_covers(measured):
    group = measured[measured["variante"] == "tapada_total"]
    assert _share(measured, "tapada_total", {"tapada"}) >= 0.90
    assert 0.9 <= group["lap_var"].median() <= 6.5  # real: 0.9 to 6.5


def test_synthetic_partial_cover_is_never_called_normal(measured):
    # The classic rules call most of them "girada" (a sharp rest of the scene that does not match the reference), so
    # this class is evaluated as sabotage against normal, not by its own label.
    assert _share(measured, "tapada_parcial", {"normal"}) <= 0.10
