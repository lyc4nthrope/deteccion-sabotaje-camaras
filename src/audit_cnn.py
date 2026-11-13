"""Robustness audit of the CNN on the same degraded test set as the classic module (T71).

Run from the repository root after python -m src.audit and a trained CNN run:

    python -m src.audit_cnn --run lr1e-2_b32_frozen

Every test image goes through exactly the same degradation as in src/audit.py (resize to 640x360 first, then the
perturbation with the same seed per image, perturbation and level), so both models are compared on identical
images. The degraded 640x360 frame is then reduced to the CNN input (224x224, RGB, [-1, 1]), as cnn.load_images
does with the originals. The sabotage score for the ROC is 1 - P(normal), as in src/evaluate_cnn.py.

Images are processed one at a time (its 19 conditions in one batch), so memory stays low.
Outputs in results/auditoria/: predicciones_cnn_test_degradado.csv and the CNN rows of tabla.csv (the classic rows
already there are kept).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import audit, cnn, data, features, metrics

PREDICTIONS = audit.OUT_DIR / "predicciones_cnn_test_degradado.csv"
TABLE = audit.OUT_DIR / "tabla.csv"
DEFAULT_RUN = "lr1e-2_b32_frozen"


def to_cnn_input(frame: np.ndarray) -> np.ndarray:
    """A BGR frame (any size) as one CNN input: 224x224, RGB, scaled to [-1, 1] (same steps as cnn.load_images)."""
    small = cv2.resize(frame, (cnn.IMAGE_SIZE, cnn.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return small[..., ::-1].astype(np.float32) / 127.5 - 1.0


def degraded_predictions(model, images: pd.DataFrame, root: Path = data.BREMEN_ROOT) -> pd.DataFrame:
    """Class probabilities of the CNN for every image under every condition of audit.conditions()."""
    rows = []
    conditions = audit.conditions()
    for path, label in zip(images["path"], images["label"]):
        image = features.load_image(root / path)
        batch = np.stack([to_cnn_input(audit.degrade(image, name, level, path)) for name, level in conditions])
        proba = np.asarray(model.predict(batch, verbose=0))
        for (name, level), p in zip(conditions, proba):
            rows.append(
                {
                    "path": path,
                    "label": label,
                    "perturbacion": name,
                    "nivel": level,
                    "pred": cnn.CLASSES[int(np.argmax(p))],
                    **{f"p_{c}": float(v) for c, v in zip(cnn.CLASSES, p)},
                }
            )
    return pd.DataFrame(rows)


def condition_table(predictions: pd.DataFrame, model_name: str = "cnn") -> pd.DataFrame:
    """Rubric metrics per condition, with the same columns and order as audit.condition_table."""
    order = {condition: i for i, condition in enumerate(audit.conditions())}
    rows = []
    for (name, level), group in predictions.groupby(["perturbacion", "nivel"], sort=False):
        scores = 1.0 - group["p_normal"].to_numpy()
        counts = group["pred"].value_counts()
        rows.append(
            {
                "modelo": model_name,
                "perturbacion": name,
                "nivel": int(level),
                "n": len(group),
                **metrics.summary(group["label"], group["pred"], scores),
                **{f"pred_{label}": int(counts.get(label, 0)) for label in metrics.LABELS},
            }
        )
    table = pd.DataFrame(rows)
    table["_order"] = [order[(n, l)] for n, l in zip(table["perturbacion"], table["nivel"])]
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def time_per_condition(model, images: pd.DataFrame, root: Path = data.BREMEN_ROOT) -> pd.DataFrame:
    """Mean time per frame of resize + CNN on a sample of degraded images, per condition (as audit.py does)."""
    sample = images.sample(audit.TIME_SAMPLE, random_state=0)
    originals = {p: features.load_image(root / p) for p in sample["path"]}
    rows = []
    for name, level in audit.conditions():
        frames = [audit.degrade(originals[p], name, level, p) for p in sample["path"]]
        timing = metrics.time_per_image(lambda f: model.predict(to_cnn_input(f)[None], verbose=0), frames)
        rows.append({"perturbacion": name, "nivel": level, "ms_por_cuadro": timing["mean_ms"]})
    return pd.DataFrame(rows)


def merge_into_table(existing: pd.DataFrame, cnn_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep every row of the other models and replace the CNN rows."""
    others = existing[existing["modelo"] != cnn_rows["modelo"].iloc[0]]
    return pd.concat([others, cnn_rows[existing.columns]], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN, help="folder in results/cnn/ with model.keras")
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(data.RESULTS_DIR / "cnn" / args.run / "model.keras")
    split = data.load_split()
    test = split[split["set"] == "test"].reset_index(drop=True)

    predictions = degraded_predictions(model, test)
    predictions.to_csv(PREDICTIONS, index=False)
    table = condition_table(predictions).merge(time_per_condition(model, test), on=["perturbacion", "nivel"])
    merged = merge_into_table(pd.read_csv(TABLE), table)
    merged.to_csv(TABLE, index=False)
    print(table.drop(columns=[c for c in table.columns if c.startswith("pred_")]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
