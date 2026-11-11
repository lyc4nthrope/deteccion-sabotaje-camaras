"""Error analysis of the classic module (T44).

For every misclassified image, the rule of the decision tree that produced the wrong label, with the descriptor it
compared, its value, its threshold and the margin (value - threshold). The rule is read from the same thresholds and
the same branches as classic.predict, so the explanation is the cause of the decision, not a guess from the
descriptors that look unusual.

Outputs: results/clasico/errores_test.csv and results/clasico/errores_camara_no_vista.csv (one row per error),
results/clasico/errores_resumen_<conjunto>.csv (one row per true -> predicted pair and rule),
results/figuras/errores_ejemplos.png (up to 3 thumbnails per error type).
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
THUMB = (240, 135)


def _rule(name: str, outcome: str, descriptor: str, value: float, threshold: float) -> dict:
    margin = value - threshold if not np.isnan(value) else np.nan
    return {
        "regla": name,
        "resultado": outcome,
        "descriptor_decisivo": descriptor,
        "valor": round(value, 3) if not np.isnan(value) else np.nan,
        "umbral": round(threshold, 3),
        "margen": round(margin, 3) if not np.isnan(margin) else np.nan,
    }


def decisive_rule(row: pd.Series, t: classic.Thresholds) -> dict:
    """The branch of the tree that decided this image (same order as classic.predict)."""
    matches = float(row["orb_matches"])
    angle = abs(float(row["angle"])) if pd.notna(row["angle"]) else np.nan
    if matches >= t.match_min:
        if np.isnan(angle):
            return _rule("coincide_sin_angulo_fiable", "girada", "angle", np.nan, t.angle_max)
        if angle > t.angle_max:
            return _rule("coincide_pero_giro_grande", "girada", "angle", angle, t.angle_max)
        return _rule("coincide_y_giro_dentro_del_limite", "normal", "orb_matches", matches, t.match_min)
    lap = float(row["lap_var"])
    if lap >= t.blur_max:
        return _rule("sin_coincidencias_y_nitida", "girada", "lap_var", lap, t.blur_max)
    feature = float(row[t.cover_feature])
    covered = feature < t.cover_threshold if t.cover_direction == "<" else feature >= t.cover_threshold
    outcome = "tapada" if covered else "desenfocada"
    return _rule("sin_coincidencias_y_borrosa", outcome, t.cover_feature, feature, t.cover_threshold)


def analyse(descriptors: pd.DataFrame, thresholds: classic.Thresholds, split: str = "test") -> pd.DataFrame:
    data = descriptors[descriptors["set"] == split].copy().reset_index(drop=True)
    data["pred"] = classic.predict(data, thresholds)
    errors = data[data["pred"] != data["label"]]
    rows = []
    for _, row in errors.iterrows():
        rule = decisive_rule(row, thresholds)
        if rule["resultado"] != row["pred"]:
            raise ValueError(f"Rule {rule['regla']} says {rule['resultado']}, classic.predict says {row['pred']}")
        rows.append({"path": row["path"], "label": row["label"], "pred": row["pred"], **rule})
    return pd.DataFrame(rows)


def summary(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return errors
    return (errors.groupby(["label", "pred", "regla", "descriptor_decisivo"]).size()
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
    unseen = pd.read_csv(CLASSIC / "descriptores_camara_no_vista.csv").assign(set="unseen")
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_errors = []
    for label, table, split in (("test", descriptors, "test"), ("camara_no_vista", unseen, "unseen")):
        errors = analyse(table, thresholds, split)
        errors.to_csv(CLASSIC / f"errores_{label}.csv", index=False)
        summary(errors).to_csv(CLASSIC / f"errores_resumen_{label}.csv", index=False)
        all_errors.append(errors)
        print(f"{label}: {len(errors)} errors")
    cv2.imwrite(str(FIGURES / "errores_ejemplos.png"), examples(pd.concat(all_errors, ignore_index=True)))


if __name__ == "__main__":
    main()
