"""Does a correction applied BEFORE the CNN recover the attacks it fails on? (T90, T93)

The classic module recovers salt and pepper with a median filter (src/corrections_audit.py). The CNN audit
(src/audit_cnn.py) only measured the CNN on the raw degraded frames, so nobody knew whether the same filter helps it.
This module feeds the CNN the frame the demo would give it with the correction key on: degrade, then correct, then
reduce to the network input. The test images and the per-image seeds are the ones of the other audits, so the
unfiltered side is read from results/auditoria/predicciones_cnn_test_degradado.csv.

    python -m src.audit_cnn_filter --run lr1e-2_b32_frozen

Output: results/auditoria/cnn_con_filtro.csv, one row per condition with the score before and after.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import audit, audit_cnn, cnn, corrections, data, features, metrics

OUT = audit.OUT_DIR / "cnn_con_filtro.csv"
CORRECTION = "mediana"
# The attacks where the CNN loses the most (sección 6 de docs/defensa.md) plus the clean frames, to see the cost.
PLAN = [audit.ORIGINAL] + [(name, level) for name in ("sal_pimienta", "jpeg", "ruido_gaussiano") for level in (1, 2, 3)]


def filtered_predictions(model, images: pd.DataFrame, plan: list[tuple[str, int]], correction: str,
                         root: Path = data.BREMEN_ROOT) -> pd.DataFrame:
    """CNN class probabilities for every image under every condition of the plan, correcting before the network."""
    if correction not in corrections.CORRECTIONS:
        raise ValueError(f"unknown correction {correction!r}; expected one of {tuple(corrections.CORRECTIONS)}")
    fix = corrections.CORRECTIONS[correction]
    rows = []
    for path, label in zip(images["path"], images["label"]):
        image = features.load_image(root / path)
        batch = np.stack([audit_cnn.to_cnn_input(fix(audit.degrade(image, name, level, path))) for name, level in plan])
        proba = np.asarray(model.predict(batch, verbose=0))
        for (name, level), p in zip(plan, proba):
            rows.append(
                {
                    "path": path,
                    "label": label,
                    "perturbacion": name,
                    "nivel": level,
                    "correccion": correction,
                    "pred": cnn.CLASSES[int(np.argmax(p))],
                    **{f"p_{c}": float(v) for c, v in zip(cnn.CLASSES, p)},
                }
            )
    return pd.DataFrame(rows)


def _scores(group: pd.DataFrame) -> dict:
    return metrics.summary(group["label"], group["pred"], 1.0 - group["p_normal"].to_numpy())


def before_after(unfiltered: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    """Per condition of `filtered`: the metrics of the same images without and with the correction."""
    rows = []
    for (name, level), after in filtered.groupby(["perturbacion", "nivel"], sort=False):
        before = unfiltered[(unfiltered["perturbacion"] == name) & (unfiltered["nivel"] == level)]
        b, a = _scores(before), _scores(after)
        rows.append(
            {
                "perturbacion": name,
                "nivel": int(level),
                "correccion": after["correccion"].iloc[0],
                "n": len(after),
                **{f"{k}_antes": b[k] for k in ("accuracy", "f1_macro", "fpr_normal")},
                **{f"{k}_despues": a[k] for k in ("accuracy", "f1_macro", "fpr_normal")},
            }
        )
    table = pd.DataFrame(rows)
    table = table.rename(columns={"f1_macro_antes": "f1_antes", "f1_macro_despues": "f1_despues"})
    table["delta_f1"] = table["f1_despues"] - table["f1_antes"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=audit_cnn.DEFAULT_RUN, help="folder in results/cnn/ with model.keras")
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(data.RESULTS_DIR / "cnn" / args.run / "model.keras")
    split = data.load_split()
    test = split[split["set"] == "test"].reset_index(drop=True)

    filtered = filtered_predictions(model, test, PLAN, CORRECTION)
    unfiltered = pd.read_csv(audit_cnn.PREDICTIONS)
    table = before_after(unfiltered, filtered)
    table.to_csv(OUT, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
