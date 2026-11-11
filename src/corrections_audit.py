"""Before/after evaluation of the correction filters on the degraded test set (T92).

Run from the repository root after python -m src.audit:

    python -m src.corrections_audit

Each perturbation is paired with the filters that the theory proposes against it (median for impulse noise,
inverse gamma or CLAHE for dark scenes, and so on). Every filter is also applied to the clean images: in the
live demo the attack is not known in advance, so a filter must not damage normal frames. The correction is
applied after the degradation, on the 640x360 image, and then the classic descriptors and rules run as usual.
Outputs in results/auditoria/: correcciones.csv and correcciones_antes_despues.csv. With --saturacion it only
measures how many pixels the lightening clips at 255 (saturacion_brillo_claro.csv), which explains why its inverse
cannot recover the image.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import audit, classic, corrections, data, features

NONE = "ninguna"

# Filters proposed for each perturbation. A name ending in "_inverso" is level specific and expands to
# "<name>_nivel_<level>" (the inverse of that exact level).
PAIRED = {
    "brillo_oscuro": ["clahe"],
    "brillo_claro": ["brillo_claro_inverso"],
    "gamma": ["gamma_inverso", "clahe"],
    "ruido_gaussiano": ["mediana", "bilateral", "gaussiano"],
    "sal_pimienta": ["mediana"],
    "jpeg": ["bilateral"],
}
LEVEL_SPECIFIC = ("gamma_inverso", "brillo_claro_inverso")

# Level-independent filters, also applied to the clean images to check they do no harm.
CLEAN_CHECK = ["clahe", "mediana", "gaussiano", "bilateral"]

# Fixed sample of test images used to measure how much the lightening saturates (T92, brillo claro).
SATURATION_SAMPLE = 40

ALL_CORRECTIONS = sorted(
    {f"{name}_nivel_{level}" for name in LEVEL_SPECIFIC for level in audit.P.LEVELS} | set(CLEAN_CHECK)
)


def corrections_for(perturbation: str, level: int, available: set[str]) -> list[str]:
    """Paired correction names for one condition, keeping only the filters that exist."""
    names = [f"{name}_nivel_{level}" if name in LEVEL_SPECIFIC else name for name in PAIRED[perturbation]]
    return [name for name in names if name in available]


def runs(available: set[str]) -> list[tuple[str, int, str]]:
    """(perturbation, level, correction) to evaluate: every condition without correction, plus its pairs."""
    result = []
    for perturbation, level in audit.conditions():
        result.append((perturbation, level, NONE))
        if (perturbation, level) == audit.ORIGINAL:
            result += [(perturbation, level, name) for name in CLEAN_CHECK if name in available]
        else:
            result += [(perturbation, level, name) for name in corrections_for(perturbation, level, available)]
    return result


def correct(image, perturbation: str, level: int, correction: str, path: str):
    """Degrade one image (resize first, seeded perturbation) and then apply the correction filter."""
    degraded = audit.degrade(image, perturbation, level, path)
    if correction == NONE:
        return degraded
    return corrections.CORRECTIONS[correction](degraded)


def corrected_descriptors(images: pd.DataFrame, matcher: features.ReferenceMatcher, plan) -> pd.DataFrame:
    """Descriptor vector of every image under every (condition, correction) of the plan."""
    rows = []
    for path, label in zip(images["path"], images["label"]):
        image = features.load_image(data.BREMEN_ROOT / path)
        for perturbation, level, correction in plan:
            vector = features.extract(correct(image, perturbation, level, correction, path), matcher)
            rows.append({"path": path, "label": label, "perturbacion": perturbation, "nivel": level,
                         "correccion": correction, **vector})
    return pd.DataFrame(rows)


def before_after(table: pd.DataFrame) -> pd.DataFrame:
    """One row per correction with the metrics of its condition before (no filter) and after."""
    keys = ["perturbacion", "nivel"]
    base = table[table["correccion"] == NONE].set_index(keys)
    rows = []
    for _, row in table[table["correccion"] != NONE].iterrows():
        before = base.loc[(row["perturbacion"], row["nivel"])]
        rows.append({
            "perturbacion": row["perturbacion"], "nivel": int(row["nivel"]), "correccion": row["correccion"],
            "f1_antes": before["f1_macro"], "f1_despues": row["f1_macro"],
            "mejora_f1": row["f1_macro"] - before["f1_macro"],
            "fpr_antes": before["fpr_normal"], "fpr_despues": row["fpr_normal"],
        })
    return pd.DataFrame(rows)


def saturation(image: np.ndarray, level: int, path: str = "") -> dict:
    """Why the inverse of the lightening cannot work: share of pixels the attack clips at 255, and how far the
    inverse leaves the image from the original, overall and on the clipped pixels (gray levels, mean of channels)."""
    small = audit.resize(image)
    lit = audit.degrade(image, "brillo_claro", level, path)
    clipped = (lit == 255).any(axis=2)
    error = np.abs(corrections.unbrighten(lit, level).astype(int) - small.astype(int)).mean(axis=2)
    return {
        "nivel": level,
        "saturados_original": float((small == 255).any(axis=2).mean()),
        "saturados": float(clipped.mean()),
        "error_medio": float(error.mean()),
        "error_en_saturados": float(error[clipped].mean()) if clipped.any() else 0.0,
    }


def saturation_table(images: pd.DataFrame, sample: int = SATURATION_SAMPLE, seed: int = 0) -> pd.DataFrame:
    """Mean of saturation() over a fixed sample of images, one row per lightening level."""
    chosen = images.sample(min(sample, len(images)), random_state=seed)
    rows = []
    for path in chosen["path"]:
        image = features.load_image(data.BREMEN_ROOT / path)
        rows += [saturation(image, level, path) for level in audit.P.LEVELS]
    return pd.DataFrame(rows).groupby("nivel", as_index=False).mean().assign(n=len(chosen))


def main() -> None:
    audit.OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = data.load_split()
    test = split[split["set"] == "test"].reset_index(drop=True)
    references = pd.read_csv(data.SPLITS_DIR / "referencias.csv")["path"]
    matcher = features.ReferenceMatcher(
        [features.preprocess(features.load_image(data.BREMEN_ROOT / p))["gray"] for p in references]
    )
    thresholds = classic.load_thresholds(data.RESULTS_DIR / "clasico" / "umbrales.json")
    plan = runs(set(corrections.CORRECTIONS))
    missing = sorted(set(ALL_CORRECTIONS) - set(corrections.CORRECTIONS))
    if missing:
        print("Correcciones aún no disponibles (se omiten):", ", ".join(missing))

    descriptors = corrected_descriptors(test, matcher, plan)
    tables = []
    for correction, group in descriptors.groupby("correccion", sort=False):
        table = audit.condition_table(group, thresholds, model="clasico")
        tables.append(table.assign(correccion=correction))
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(audit.OUT_DIR / "correcciones.csv", index=False)
    summary = before_after(full)
    summary.to_csv(audit.OUT_DIR / "correcciones_antes_despues.csv", index=False)
    print(summary.round(3).to_string(index=False))


def main_saturation() -> None:
    split = data.load_split()
    table = saturation_table(split[split["set"] == "test"].reset_index(drop=True))
    table.to_csv(audit.OUT_DIR / "saturacion_brillo_claro.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saturacion", action="store_true",
                        help="only measure the pixels clipped by the lightening (about 30 s)")
    if parser.parse_args().saturacion:
        main_saturation()
    else:
        main()
