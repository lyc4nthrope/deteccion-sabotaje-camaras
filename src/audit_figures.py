"""Figures and tables of the robustness audit comparing both models (T72, T73).

Run from the repository root after python -m src.audit and python -m src.audit_cnn:

    python -m src.audit_figures

Reads the per-image predictions already computed (nothing is re-run):
- classic: results/auditoria/descriptores_test_degradado.csv + the thresholds of results/clasico/umbrales.json;
- CNN: results/auditoria/predicciones_cnn_test_degradado.csv.

Outputs:
- T72: results/auditoria/matrices_cruzadas.csv and results/figuras/auditoria_matrices.png, confusion matrix of each
  model under each critical condition (the original and the strongest level of each perturbation).
- T73: results/figuras/auditoria_roc.png (ROC of normal vs sabotage per perturbation, level 3 against the original)
  and results/figuras/auditoria_f1_severidad.png (macro F1 against the level, from results/auditoria/tabla.csv).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import audit, classic, data, metrics  # noqa: E402
from src import perturbations as P  # noqa: E402

FIGURES = data.RESULTS_DIR / "figuras"
CRITICAL = [audit.ORIGINAL] + [(name, 3) for name in P.PERTURBATIONS]
MODELS = {"clasico": "Clásico", "cnn": "CNN"}
NAMES = {
    "original": "original",
    "brillo_oscuro": "brillo oscuro",
    "brillo_claro": "brillo claro",
    "gamma": "gamma",
    "ruido_gaussiano": "ruido gaussiano",
    "sal_pimienta": "sal y pimienta",
    "jpeg": "JPEG",
}


def predictions(descriptors: pd.DataFrame, thresholds: classic.Thresholds, cnn: pd.DataFrame) -> pd.DataFrame:
    """One row per model, image and condition: true label, predicted label and sabotage score."""
    keys = ["path", "label", "perturbacion", "nivel"]
    clasico = descriptors[keys].assign(
        modelo="clasico",
        pred=classic.predict(descriptors, thresholds),
        score=classic.sabotage_score(descriptors, thresholds),
    )
    red = cnn[keys].assign(modelo="cnn", pred=cnn["pred"].to_numpy(), score=1.0 - cnn["p_normal"].to_numpy())
    return pd.concat([clasico, red], ignore_index=True)


def cross_matrices(frame: pd.DataFrame, conditions=CRITICAL) -> pd.DataFrame:
    """Long table of confusion counts (model, condition, true, predicted, n) for the given conditions (T72)."""
    rows = []
    for model in MODELS:
        for name, level in conditions:
            group = frame[(frame["modelo"] == model) & (frame["perturbacion"] == name) & (frame["nivel"] == level)]
            matrix = metrics.confusion(group["label"], group["pred"])
            for true in matrix.index:
                for pred in matrix.columns:
                    rows.append({"modelo": model, "perturbacion": name, "nivel": level, "real": true,
                                 "predicha": pred, "n": int(matrix.loc[true, pred])})
    return pd.DataFrame(rows)


def plot_matrices(table: pd.DataFrame, conditions=CRITICAL):
    fig, axes = plt.subplots(len(MODELS), len(conditions), figsize=(3.1 * len(conditions), 6.4))
    for r, model in enumerate(MODELS):
        for c, (name, level) in enumerate(conditions):
            cell = table[(table["modelo"] == model) & (table["perturbacion"] == name) & (table["nivel"] == level)]
            matrix = cell.pivot(index="real", columns="predicha", values="n").loc[metrics.LABELS, metrics.LABELS]
            ax = axes[r, c]
            ax.imshow(matrix.to_numpy(), cmap="Blues")
            for i in range(len(metrics.LABELS)):
                for j in range(len(metrics.LABELS)):
                    value = int(matrix.iat[i, j])
                    color = "white" if value > matrix.to_numpy().max() / 2 else "black"
                    ax.text(j, i, value, ha="center", va="center", fontsize=8, color=color)
            ax.set_xticks(range(4), [l[:5] for l in metrics.LABELS], fontsize=7)
            ax.set_yticks(range(4), [l[:5] for l in metrics.LABELS], fontsize=7)
            ax.set_title(f"{MODELS[model]}: {NAMES[name]}" + (f" {level}" if level else ""), fontsize=9)
    fig.supxlabel("Clase predicha")
    fig.supylabel("Clase real")
    fig.tight_layout()
    return fig


def plot_roc(frame: pd.DataFrame):
    """ROC of normal vs sabotage per perturbation: level 3 (solid) against the original (dashed), both models."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex=True, sharey=True)
    styles = {"clasico": "tab:orange", "cnn": "tab:blue"}
    for ax, name in zip(axes.ravel(), P.PERTURBATIONS):
        for model, color in styles.items():
            for (cond, level), line in ((audit.ORIGINAL, "--"), ((name, 3), "-")):
                group = frame[(frame["modelo"] == model) & (frame["perturbacion"] == cond) & (frame["nivel"] == level)]
                roc = metrics.binary_roc(group["label"], group["score"])
                label = f"{MODELS[model]} {'original' if level == 0 else 'nivel 3'} (AUC {roc['auc']:.2f})"
                ax.plot(roc["fpr"], roc["tpr"], line, color=color, label=label, linewidth=1.4)
        ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
        ax.set_title(NAMES[name])
        ax.legend(fontsize=7, loc="lower right")
    fig.supxlabel("Tasa de falsas alarmas (normales marcadas como sabotaje)")
    fig.supylabel("Tasa de detección de sabotaje")
    fig.tight_layout()
    return fig


def f1_by_level(table: pd.DataFrame) -> pd.DataFrame:
    """Macro F1 per model, perturbation and level 0-3; level 0 is the original image (T73)."""
    original = table[table["perturbacion"] == "original"]
    rows = []
    for model in MODELS:
        base = float(original.loc[original["modelo"] == model, "f1_macro"].iloc[0])
        for name in P.PERTURBATIONS:
            rows.append({"modelo": model, "perturbacion": name, "nivel": 0, "f1_macro": base})
            part = table[(table["modelo"] == model) & (table["perturbacion"] == name)]
            rows += [{"modelo": model, "perturbacion": name, "nivel": int(l), "f1_macro": float(f)}
                     for l, f in zip(part["nivel"], part["f1_macro"])]
    return pd.DataFrame(rows).sort_values(["modelo", "perturbacion", "nivel"]).reset_index(drop=True)


def plot_f1(levels: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), P.PERTURBATIONS):
        for model, color in (("clasico", "tab:orange"), ("cnn", "tab:blue")):
            part = levels[(levels["modelo"] == model) & (levels["perturbacion"] == name)]
            ax.plot(part["nivel"], part["f1_macro"], "o-", color=color, label=MODELS[model])
        ax.set_title(NAMES[name])
        ax.set_xticks([0, 1, 2, 3], ["orig.", "1", "2", "3"])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.supxlabel("Nivel de la perturbación")
    fig.supylabel("F1 macro")
    fig.tight_layout()
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    thresholds = classic.load_thresholds(data.RESULTS_DIR / "clasico" / "umbrales.json")
    frame = predictions(
        pd.read_csv(audit.OUT_DIR / "descriptores_test_degradado.csv"),
        thresholds,
        pd.read_csv(audit.OUT_DIR / "predicciones_cnn_test_degradado.csv"),
    )
    matrices = cross_matrices(frame)
    matrices.to_csv(audit.OUT_DIR / "matrices_cruzadas.csv", index=False)
    plot_matrices(matrices).savefig(FIGURES / "auditoria_matrices.png", dpi=110)
    plot_roc(frame).savefig(FIGURES / "auditoria_roc.png", dpi=110)
    plot_f1(f1_by_level(pd.read_csv(audit.OUT_DIR / "tabla.csv"))).savefig(FIGURES / "auditoria_f1_severidad.png", dpi=110)
    print("matrices:", len(matrices), "filas; figuras en", FIGURES)


if __name__ == "__main__":
    main()
