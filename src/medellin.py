"""Selection of normal Medellín images for the regional cross test (T17, T17b, D12).

The camera of each image is recovered by clustering (src/medellin_groups.py, T17b); the file name (`imagenN_...`) is
not a camera id. `select_from_groups` picks about 600 images spread over the reliable camera groups, never a
reference of its group (an image compared with itself is "normal" for sure) and never one taken within
MIN_GAP_MINUTES of a reference (near duplicates), evenly spaced in time inside each group.
`select_normal` is the first version, which grouped by file name; it is kept only for its tests.

Run:  python -m src.medellin   (reads data/splits/medellin_grupos.csv, writes data/splits/medellin_normales.csv)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_GAP_MINUTES = 10  # minimum time between a selected image and any reference of its group
NAME_PATTERN = re.compile(r"^imagen(\d+)_(\d{8})_(\d{6})\.jpg$")


def add_group_columns(inventory: pd.DataFrame) -> pd.DataFrame:
    """Add `grupo` (the N of imagenN) and sort by timestamp inside each group."""
    parsed = inventory["archivo"].str.extract(NAME_PATTERN)
    if parsed.isna().any().any():
        bad = inventory.loc[parsed.isna().any(axis=1), "archivo"].tolist()[:5]
        raise ValueError(f"file names do not match the expected pattern: {bad}")
    out = inventory.copy()
    out["grupo"] = parsed[0].astype(int)
    out["instante"] = parsed[1] + parsed[2]
    return out.sort_values(["grupo", "instante"]).reset_index(drop=True)


def allocation_per_group(group_sizes: pd.Series, total: int) -> pd.Series:
    """Water-filling: give every group min(size, cap) images, with the largest cap that keeps the total <= total."""
    if total >= int(group_sizes.sum()):
        return group_sizes.copy()
    low, high = 0, int(group_sizes.max())
    while low < high:
        mid = (low + high + 1) // 2
        if int(group_sizes.clip(upper=mid).sum()) <= total:
            low = mid
        else:
            high = mid - 1
    alloc = group_sizes.clip(upper=low).astype(int)
    remainder = total - int(alloc.sum())
    # Hand the leftover images to the groups that were capped, largest first, one each.
    capped = group_sizes[group_sizes > alloc].sort_values(ascending=False).index
    for group in capped[:remainder]:
        alloc[group] += 1
    return alloc


def _spread(grouped: pd.DataFrame, group_column: str, total: int) -> pd.DataFrame:
    """About `total` rows spread over the groups (water-filling), evenly spaced in time inside each group."""
    grouped = grouped.sort_values([group_column, "instante"]).reset_index(drop=True)
    sizes = grouped.groupby(group_column).size()
    alloc = allocation_per_group(sizes, total)
    picked = []
    for group, n in alloc.items():
        rows = grouped[grouped[group_column] == group]
        if n >= len(rows):
            picked.append(rows)
        else:
            positions = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
            picked.append(rows.iloc[sorted(set(positions))])
    return pd.concat(picked).sort_values([group_column, "instante"]).reset_index(drop=True)


def select_normal(inventory: pd.DataFrame, total: int = 600) -> pd.DataFrame:
    """First version: about `total` images spread over the groups of the file name (not the camera)."""
    return _spread(add_group_columns(inventory), "grupo", total)


def select_from_groups(groups: pd.DataFrame, total: int = 600, min_gap_minutes: int = MIN_GAP_MINUTES) -> pd.DataFrame:
    """About `total` images of the reliable camera groups that are not references and are far from them in time.

    `groups` is data/splits/medellin_grupos.csv (columns cluster, instante, es_referencia, confiable, ...). The
    result has the cluster as `grupo` and keeps the file-name group as `grupo_nombre` for traceability.
    """
    moments = pd.to_datetime(groups["instante"].astype(str), format="%Y%m%d%H%M%S")
    refs = groups["es_referencia"]
    ref_moments = {c: moments[refs & (groups["cluster"] == c)].to_numpy() for c in groups["cluster"].unique()}
    gap = np.timedelta64(int(min_gap_minutes * 60), "s")
    far = np.array(
        [
            len(ref_moments[c]) == 0 or np.abs(ref_moments[c] - np.datetime64(m)).min() >= gap
            for c, m in zip(groups["cluster"], moments)
        ]
    )
    candidates = groups[groups["confiable"] & ~refs & far].copy()
    candidates["grupo"] = candidates["cluster"]
    candidates["instante"] = candidates["instante"].astype(str)
    chosen = _spread(candidates, "grupo", total)
    return chosen[["archivo", "grupo", "instante", "grupo_nombre"]]


def main() -> None:
    from src import data

    groups = pd.read_csv(data.SPLITS_DIR / "medellin_grupos.csv", dtype={"instante": str})
    selection = select_from_groups(groups)
    selection.to_csv(data.SPLITS_DIR / "medellin_normales.csv", index=False)
    print(f"{len(selection)} images in {selection['grupo'].nunique()} camera groups")


if __name__ == "__main__":
    main()
