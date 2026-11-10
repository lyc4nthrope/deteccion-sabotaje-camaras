"""Compare ORB, SIFT and KAZE as the scene-matching descriptor of the classic module (T38).

Run from the repository root:

    python -m src.compare_descriptors

For each detector, the match threshold that separates normal from sabotaged images is calibrated on a train
sample (balanced accuracy), and the detector is evaluated on the whole val set. Test is not used.
Output: results/clasico/comparacion_descriptores.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import classic, data, features, metrics

OUT = data.RESULTS_DIR / "clasico" / "comparacion_descriptores.csv"
TRAIN_PER_CLASS = 50
METHODS = ("orb", "sift", "kaze")


def _load_gray(path: str) -> np.ndarray:
    return features.preprocess(features.load_image(data.BREMEN_ROOT / path))["gray"]


def compare(train: pd.DataFrame, val: pd.DataFrame, references: pd.Series) -> pd.DataFrame:
    """One row per detector with its calibrated threshold, val metrics and time per frame."""
    reference_grays = [_load_gray(p) for p in references]
    train_grays = [_load_gray(p) for p in train["path"]]
    val_grays = [_load_gray(p) for p in val["path"]]
    rows = []
    for method in METHODS:
        matcher = features.ReferenceMatcher(reference_grays, method=method)
        train_matches = np.array([matcher.match(g)["orb_matches"] for g in train_grays])
        threshold, _, _ = classic.best_threshold(train_matches, train["label"] == "normal", directions=(">=",))

        val_matches = np.array([matcher.match(g)["orb_matches"] for g in val_grays])
        predicted = np.where(val_matches >= threshold, "normal", "saboteada")
        sabotage = val["label"].to_numpy() != "normal"
        girada = val["label"].to_numpy() == "girada"
        timing = metrics.time_per_image(matcher.match, val_grays[:100])
        rows.append(
            {
                "detector": method,
                "umbral_coincidencias": threshold,
                "exactitud_balanceada_val": float(
                    (np.mean(predicted[sabotage] == "saboteada") + np.mean(predicted[~sabotage] == "normal")) / 2
                ),
                "falsas_alarmas_val": float(np.mean(predicted[~sabotage] == "saboteada")),
                "giradas_detectadas_val": float(np.mean(predicted[girada] == "saboteada")),
                "auc_val": metrics.binary_roc(val["label"], -val_matches)["auc"],
                "coincidencias_normal_mediana": float(np.median(val_matches[~sabotage])),
                "coincidencias_girada_mediana": float(np.median(val_matches[girada])),
                "ms_por_cuadro": timing["mean_ms"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    split = data.load_split()
    references = pd.read_csv(data.SPLITS_DIR / "referencias.csv")["path"]
    train = split[(split["set"] == "train") & ~split["path"].isin(references)]
    train = train.groupby("label", group_keys=False).sample(TRAIN_PER_CLASS, random_state=0)
    val = split[split["set"] == "val"]

    table = compare(train, val, references)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
