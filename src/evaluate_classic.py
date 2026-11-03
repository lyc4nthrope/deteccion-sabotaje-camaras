"""Calibrate the classic module on train and evaluate it on val, test and the unseen camera (T43).

Run from the repository root after python -m src.build_features:

    python -m src.evaluate_classic

Outputs in results/clasico/: umbrales.json, metricas.csv, matriz_<conjunto>.csv, roc_<conjunto>.csv, tiempo.json.
"""

from __future__ import annotations

import json

import pandas as pd

from src import classic, data, features, metrics

OUT_DIR = data.RESULTS_DIR / "clasico"
TIME_SAMPLE = 100


def evaluate(descriptors: pd.DataFrame, thresholds: classic.Thresholds, name: str) -> dict:
    """Rubric metrics of one evaluation set; writes its confusion matrix and ROC points."""
    predicted = classic.predict(descriptors, thresholds)
    scores = classic.sabotage_score(descriptors, thresholds)
    metrics.confusion(descriptors["label"], predicted).to_csv(OUT_DIR / f"matriz_{name}.csv")
    roc = metrics.binary_roc(descriptors["label"], scores)
    pd.DataFrame({"fpr": roc["fpr"], "tpr": roc["tpr"]}).to_csv(OUT_DIR / f"roc_{name}.csv", index=False)
    return {"conjunto": name, "n": len(descriptors), **metrics.summary(descriptors["label"], predicted, scores)}


def time_per_frame(paths: pd.Series, thresholds: classic.Thresholds) -> dict:
    """End-to-end time per frame (descriptors + rules) on decoded images, excluding disk reads."""
    references = pd.read_csv(data.SPLITS_DIR / "referencias.csv")
    matcher = features.ReferenceMatcher(
        [features.preprocess(features.load_image(data.BREMEN_ROOT / p))["gray"] for p in references["path"]]
    )
    images = [features.load_image(data.BREMEN_ROOT / p) for p in paths]

    def classify(image):
        return classic.predict(pd.DataFrame([features.extract(image, matcher)]), thresholds)[0]

    return metrics.time_per_image(classify, images)


def main() -> None:
    location1 = pd.read_csv(OUT_DIR / "descriptores_ubicacion1.csv")
    unseen = pd.read_csv(OUT_DIR / "descriptores_camara_no_vista.csv")

    thresholds = classic.calibrate(location1[location1["set"] == "train"])
    classic.save_thresholds(thresholds, OUT_DIR / "umbrales.json")

    rows = [evaluate(location1[location1["set"] == name], thresholds, name) for name in ("train", "val", "test")]
    rows.append(evaluate(unseen, thresholds, "camara_no_vista"))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "metricas.csv", index=False)

    test_paths = location1[location1["set"] == "test"]["path"].sample(TIME_SAMPLE, random_state=0)
    timing = time_per_frame(test_paths, thresholds)
    (OUT_DIR / "tiempo.json").write_text(json.dumps(timing, indent=2))

    print("Umbrales:", thresholds)
    print(table.round(3).to_string(index=False))
    print("Tiempo por frame:", {k: round(v, 1) for k, v in timing.items()})
    for name in ("test", "camara_no_vista"):
        print(f"\nMatriz de confusión ({name}), filas = real, columnas = predicho:")
        print(pd.read_csv(OUT_DIR / f"matriz_{name}.csv", index_col=0).to_string())


if __name__ == "__main__":
    main()
