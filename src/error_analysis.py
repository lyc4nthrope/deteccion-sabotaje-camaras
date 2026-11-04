"""Error analysis of the classic module on the test set (T44).

For every misclassified image: its true and predicted class, and the descriptors that moved away from the typical
value of the true class (median over the correctly classified images of that class). The result explains each error
type by the descriptor that failed (the question T44 asks).

Outputs: results/clasico/errores_test.csv (one row per error), results/clasico/errores_resumen.csv (one row per
true -> predicted pair), results/figuras/errores_ejemplos.png (up to 3 thumbnails per error type).
Run: python -m src.error_analysis
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import classic

ROOT = Path(__file__).resolve().parents[1]
CLASSIC = ROOT / "results" / "clasico"
FIGURES = ROOT / "results" / "figuras"
BREMEN = ROOT / "data" / "raw" / "bremen"
DESCRIPTORS = ["lap_var", "gray_std", "entropy", "sat_mean", "hue_std", "skin_ratio", "orb_inliers", "angle"]
THUMB = (240, 135)


def analyse(descriptors: pd.DataFrame, thresholds: classic.Thresholds, split: str = "test") -> pd.DataFrame:
    data = descriptors[descriptors["set"] == split].copy().reset_index(drop=True)
    data["pred"] = classic.predict(data, thresholds)
    correct = data[data["pred"] == data["label"]]
    typical = correct.groupby("label")[DESCRIPTORS].median()
    errors = data[data["pred"] != data["label"]].copy()
    rows = []
    for _, row in errors.iterrows():
        ref = typical.loc[row["label"]]
        moved = {d: float(row[d]) - float(ref[d]) for d in DESCRIPTORS if pd.notna(row[d]) and pd.notna(ref[d])}
        worst = max(moved, key=lambda d: abs(moved[d]) / (abs(ref[d]) + 1e-9)) if moved else ""
        rows.append({**row[["path", "label", "pred"]].to_dict(), "descriptor_principal": worst,
                     "valor": round(float(row[worst]), 3) if worst else np.nan,
                     "tipico_clase": round(float(ref[worst]), 3) if worst else np.nan})
    return pd.DataFrame(rows)


def summary(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return errors
    return (errors.groupby(["label", "pred", "descriptor_principal"]).size()
            .rename("n").reset_index().sort_values("n", ascending=False))


def examples(errors: pd.DataFrame, per_type: int = 3) -> np.ndarray:
    rows = []
    for (true, pred), group in errors.groupby(["label", "pred"]):
        images = []
        for path in group["path"].head(per_type):
            image = cv2.imread(str(BREMEN / path))
            images.append(cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA))
        while len(images) < per_type:
            images.append(np.zeros((THUMB[1], THUMB[0], 3), np.uint8))
        bar = np.full((22, THUMB[0] * per_type, 3), 40, np.uint8)
        cv2.putText(bar, f"real {true} -> predicha {pred} ({len(group)})", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([bar, np.hstack(images)]))
    return np.vstack(rows) if rows else np.zeros((1, 1, 3), np.uint8)


def main() -> None:
    thresholds = classic.load_thresholds(CLASSIC / "umbrales.json")
    descriptors = pd.read_csv(CLASSIC / "descriptores_ubicacion1.csv")
    unseen = pd.read_csv(CLASSIC / "descriptores_camara_no_vista.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for label, table, split in (("test", descriptors, "test"), ("camara_no_vista", unseen, "unseen")):
        if split == "unseen":
            table = table.assign(set="unseen")
        errors = analyse(table, thresholds, split)
        errors.to_csv(CLASSIC / f"errores_{label}.csv", index=False)
        summary(errors).to_csv(CLASSIC / f"errores_resumen_{label}.csv", index=False)
        print(f"{label}: {len(errors)} errors")
    cv2.imwrite(str(FIGURES / "errores_ejemplos.png"), examples(errors))


if __name__ == "__main__":
    main()
