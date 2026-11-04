"""Selection of normal Medellín images for the regional cross test (T17, D12).

Camera groups come from the file name (`imagenN_AAAAMMDD_HHMMSS.jpg`). The group is only a proxy for the camera:
the ORB reference views need the same camera (T33), so the selection spreads the images across groups and records
the group for each image. Groups with few images contribute all of them.
"""

from __future__ import annotations

import re

import pandas as pd

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


def select_normal(inventory: pd.DataFrame, total: int = 600) -> pd.DataFrame:
    """Pick about `total` images spread over the camera groups, evenly spaced in time inside each group."""
    grouped = add_group_columns(inventory)
    sizes = grouped.groupby("grupo").size()
    alloc = allocation_per_group(sizes, total)
    picked = []
    for group, n in alloc.items():
        rows = grouped[grouped["grupo"] == group]
        if n >= len(rows):
            picked.append(rows)
        else:
            positions = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
            picked.append(rows.iloc[sorted(set(positions))])
    return pd.concat(picked).sort_values(["grupo", "instante"]).reset_index(drop=True)
