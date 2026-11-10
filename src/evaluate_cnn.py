"""Evaluate the chosen CNN on the test set and on the unseen camera (T64).

Uses the shared metrics of T25 (src/metrics.py), so the CNN and the classic module are compared with the same
formulas. The score for the ROC is the probability of any sabotage class (1 - P(normal)).

Outputs (results/cnn/evaluacion/): resumen.csv (one row per set), matriz_<set>.csv (confusion matrix),
tiempo.json (ms per frame, batch size 1). Run: python -m src.evaluate_cnn
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import cnn, metrics

ROOT = Path(__file__).resolve().parents[1]
UNSEEN = ROOT / "data" / "splits" / "camara_no_vista.csv"
TIMING_IMAGES = 100
DEFAULT_RUN = "lr1e-2_b32_frozen"


def predict_proba(model, images: np.ndarray) -> np.ndarray:
    return model.predict(images, batch_size=32, verbose=0)


def evaluate_set(model, table: pd.DataFrame, name: str) -> dict:
    images = cnn.load_images(table["path"].tolist())
    proba = predict_proba(model, images)
    predicted = np.array([cnn.CLASSES[i] for i in proba.argmax(axis=1)])
    truth = table["label"].to_numpy()
    scores = 1.0 - proba[:, cnn.CLASSES.index("normal")]
    row = {"conjunto": name, "n": len(table), **metrics.summary(truth, predicted, scores)}
    metrics.confusion(truth, predicted).to_csv(OUT / f"matriz_{name}.csv")
    return row


def time_frames(model, table: pd.DataFrame) -> dict:
    sample = cnn.load_images(table["path"].head(TIMING_IMAGES).tolist())
    frames = [sample[i:i + 1] for i in range(len(sample))]
    return metrics.time_per_image(lambda frame: model.predict(frame, verbose=0), frames)


def main() -> None:
    import argparse

    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="name of a folder in results/cnn/")
    args = parser.parse_args()
    run_dir = ROOT / "results" / "cnn" / args.run
    global OUT
    OUT = run_dir / "evaluacion"
    OUT.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(run_dir / "model.keras")
    test = cnn.read_split()
    test = test[test["set"] == "test"].reset_index(drop=True)
    unseen = pd.read_csv(UNSEEN)
    # Same evaluation set as the classic module: the ORB references of the unseen camera are excluded (T43, T19).
    references = pd.read_csv(ROOT / "data" / "splits" / "referencias_camara_no_vista.csv")
    unseen = unseen[~unseen["path"].isin(references["path"])].reset_index(drop=True)

    rows = [evaluate_set(model, test, "test"), evaluate_set(model, unseen, "camara_no_vista")]
    pd.DataFrame(rows).to_csv(OUT / "resumen.csv", index=False)
    timing = time_frames(model, unseen)
    (OUT / "tiempo.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
