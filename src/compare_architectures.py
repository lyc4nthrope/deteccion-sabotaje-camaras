"""Parameters and CPU time of MobileNetV2, ResNet-50 and VGG-16 for the architecture section of docs/cnn.md (T65).

The three models are built with their full ImageNet head and random weights (the weights do not change the time),
at 224x224, batch size 1. Each model is called `warmup` times (not timed) and then `runs` times per round; there are
three rounds, interleaved. The table keeps the median of the rounds and their minimum and maximum, in milliseconds:
the times depend on the machine and its load (the same code gave 83 to 116 ms for MobileNetV2), so compare the models
with each other, not with other tables.

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
ROUNDS = 3  # the whole measurement is repeated, interleaving the models, because one round varies with the machine load


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


def comparison_table(rows: list[tuple[str, int, list[float]]], repetitions: int) -> pd.DataFrame:
    """One row per (name, parameters, [median ms of each round]): median, min, max and ratio to the first model."""
    if not rows:
        raise ValueError("at least one model is needed")
    medians = [float(np.median(rounds)) for _, _, rounds in rows]
    return pd.DataFrame(
        {
            "modelo": [name for name, _, _ in rows],
            "parametros": [params for _, params, _ in rows],
            "ms_mediana": [round(m, 1) for m in medians],
            "ms_min": [round(min(r), 1) for _, _, r in rows],
            "ms_max": [round(max(r), 1) for _, _, r in rows],
            "veces_el_primero": [round(m / medians[0], 1) for m in medians],
            "repeticiones": repetitions,
            "rondas": [len(r) for _, _, r in rows],
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
    built = {name: constructor(weights=None, input_shape=shape) for name, constructor in models.items()}
    rounds = {name: [] for name in built}
    for round_number in range(ROUNDS):
        for name, model in built.items():  # interleaved: every round sees a similar machine load
            rounds[name].append(time_model(lambda x, m=model: m.predict(x, verbose=0), sample))
            print(f"round {round_number + 1} {name}: {rounds[name][-1]:.1f} ms", flush=True)
    rows = [(name, int(model.count_params()), rounds[name]) for name, model in built.items()]
    table = comparison_table(rows, RUNS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
