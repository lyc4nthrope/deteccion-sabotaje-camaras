"""Parameters and CPU time of MobileNetV2, ResNet-50 and VGG-16 for the architecture section of docs/cnn.md (T65).

The three models are built with their full ImageNet head and random weights (the weights do not change the time),
at 224x224, batch size 1. Each model is called `warmup` times (not timed) and then `runs` times; the table keeps the
median in milliseconds. The times depend on the machine: compare the models with each other, not with other tables.

Run:  python -m src.compare_architectures
Writes results/cnn/arquitecturas.csv. AlexNet is not in Keras, so it is not measured (about 61 million parameters
in the literature).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from src import data

OUT = data.RESULTS_DIR / "cnn" / "arquitecturas.csv"
WARMUP = 5
RUNS = 30


def time_model(predict: Callable, sample: np.ndarray, warmup: int = WARMUP, runs: int = RUNS,
               timer: Callable[[], float] = time.perf_counter) -> float:
    """Median time of predict(sample) in milliseconds over `runs` timed calls, after `warmup` untimed ones."""
    for _ in range(warmup):
        predict(sample)
    times = []
    for _ in range(runs):
        start = timer()
        predict(sample)
        times.append((timer() - start) * 1000.0)
    return float(np.median(times))


def comparison_table(rows: list[tuple[str, int, float]], repetitions: int) -> pd.DataFrame:
    """One row per (name, parameters, median ms), with the time as a multiple of the first model."""
    if not rows:
        raise ValueError("at least one model is needed")
    base = rows[0][2]
    return pd.DataFrame(
        {
            "modelo": [name for name, _, _ in rows],
            "parametros": [params for _, params, _ in rows],
            "ms_mediana": [round(ms, 1) for _, _, ms in rows],
            "veces_el_primero": [round(ms / base, 1) for _, _, ms in rows],
            "repeticiones": repetitions,
        }
    )


def main() -> None:
    import tensorflow as tf

    shape = (224, 224, 3)
    models = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "ResNet-50": tf.keras.applications.ResNet50,
        "VGG-16": tf.keras.applications.VGG16,
    }
    sample = np.random.default_rng(0).random((1,) + shape).astype("float32")
    rows = []
    for name, constructor in models.items():
        model = constructor(weights=None, input_shape=shape)
        ms = time_model(lambda x: model.predict(x, verbose=0), sample)
        rows.append((name, int(model.count_params()), ms))
        print(f"{name}: {rows[-1][1]} parameters, {ms:.1f} ms", flush=True)
    table = comparison_table(rows, RUNS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
