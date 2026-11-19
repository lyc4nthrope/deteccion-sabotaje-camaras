"""Cross test Bremen -> Medellín: both modules on real CCTV cameras of another country (T76).

Medellín has no sabotage, so each normal image comes with four synthetic variants (src/synthetic_tampering.py). The
classic module keeps the thresholds calibrated on Bremen (results/clasico/umbrales.json) and only changes its
references: the 8 normal images of each Medellín camera group, as it would on a new installation. The CNN is the
Bremen model, untouched. Nothing is calibrated or trained here, so the test is not tuned on Medellín.

    python -m src.evaluate_medellin --run lr1e-2_b32_frozen

Outputs in results/medellin/: predicciones.csv, metricas.csv, recall_por_variante.csv, matriz_<modelo>.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import audit_cnn, classic, cnn, data, features, metrics

IMAGES = data.ROOT / "data" / "raw" / "medellin" / "images"
SYNTHETIC = data.ROOT / "data" / "medellin_sintetico"
OUT_DIR = data.RESULTS_DIR / "medellin"
VARIANTS = ("desenfocada", "girada", "tapada_total", "tapada_parcial")
CLASS_OF = {"normal": "normal", "desenfocada": "desenfocada", "girada": "girada",
            "tapada_total": "tapada", "tapada_parcial": "tapada"}


def build_cases(normales: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per image to classify: each normal image and each of its synthetic variants, with the Bremen class."""
    unknown = set(manifest["archivo"]) - set(normales["archivo"])
    if unknown:
        raise ValueError(f"variants of images not in the normal list: {sorted(unknown)[:3]}")
    rows = [
        {"archivo": a, "grupo": g, "variante": "normal", "label": "normal", "path": IMAGES / a}
        for a, g in zip(normales["archivo"], normales["grupo"])
    ]
    for archivo, grupo, variante in zip(manifest["archivo"], manifest["grupo"], manifest["etiqueta"]):
        rows.append({"archivo": archivo, "grupo": grupo, "variante": variante, "label": CLASS_OF[variante],
                     "path": SYNTHETIC / variante / archivo})
    return pd.DataFrame(rows)


def recall_by_variant(cases: pd.DataFrame) -> pd.DataFrame:
    """Per variant: share in its expected class (`recall`) and share flagged as anything but normal (`detectada`).

    For `normal`, `recall` is the true negative rate and `detectada` is the false alarm rate.
    """
    rows = [
        {"variante": v, "n": len(g), "recall": float(np.mean(g["pred"] == g["label"])),
         "detectada": float(np.mean(g["pred"] != "normal"))}
        for v, g in cases.groupby("variante", sort=False)
    ]
    return pd.DataFrame(rows)


def classic_predictions(cases: pd.DataFrame, references: pd.DataFrame, thresholds: classic.Thresholds) -> pd.DataFrame:
    """Classic prediction with the references of each image's own camera group."""
    out = cases.copy()
    out["pred"], out["score"] = "", np.nan
    for group, part in cases.groupby("grupo"):
        refs = references[references["cluster"] == group]["archivo"]
        matcher = features.ReferenceMatcher([features.preprocess(features.load_image(IMAGES / a))["gray"] for a in refs])
        rows = pd.DataFrame([features.extract(features.load_image(p), matcher) for p in part["path"]], index=part.index)
        out.loc[part.index, "pred"] = classic.predict(rows, thresholds)
        out.loc[part.index, "score"] = classic.sabotage_score(rows, thresholds)
    return out


def cnn_predictions(cases: pd.DataFrame, model) -> pd.DataFrame:
    out = cases.copy()
    batch = np.stack([audit_cnn.to_cnn_input(features.load_image(p)) for p in cases["path"]])
    proba = np.asarray(model.predict(batch, batch_size=32, verbose=0))
    out["pred"] = [cnn.CLASSES[i] for i in proba.argmax(axis=1)]
    out["score"] = 1.0 - proba[:, cnn.CLASSES.index("normal")]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=audit_cnn.DEFAULT_RUN, help="folder in results/cnn/ with model.keras")
    args = parser.parse_args()

    import tensorflow as tf

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    normales = pd.read_csv(data.SPLITS_DIR / "medellin_normales.csv")
    manifest = pd.read_csv(SYNTHETIC / "manifest.csv")
    groups = pd.read_csv(data.SPLITS_DIR / "medellin_grupos.csv", dtype={"instante": str})
    references = groups[groups["es_referencia"]]
    thresholds = classic.load_thresholds(data.RESULTS_DIR / "clasico" / "umbrales.json")
    cases = build_cases(normales, manifest)

    model = tf.keras.models.load_model(data.RESULTS_DIR / "cnn" / args.run / "model.keras")
    results = {"clasico": classic_predictions(cases, references, thresholds), "cnn": cnn_predictions(cases, model)}

    rows, recalls = [], []
    for name, table in results.items():
        metrics.confusion(table["label"], table["pred"]).to_csv(OUT_DIR / f"matriz_{name}.csv")
        rows.append({"modelo": name, "n": len(table),
                     **metrics.summary(table["label"], table["pred"], table["score"])})
        recalls.append(recall_by_variant(table).assign(modelo=name))
    pd.concat([t.assign(modelo=n) for n, t in results.items()]).drop(columns="path").to_csv(
        OUT_DIR / "predicciones.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "metricas.csv", index=False)
    pd.concat(recalls).to_csv(OUT_DIR / "recall_por_variante.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.concat(recalls).pivot(index="variante", columns="modelo", values="recall").round(3).to_string())


if __name__ == "__main__":
    main()
