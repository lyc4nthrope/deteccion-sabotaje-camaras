"""Recover the camera of each Medellín image with unsupervised clustering (T17b, D15, class 20).

The file name (`imagenN_...`) is not a reliable camera id (55 groups, 22 with 1 to 4 images), but the classic module
needs reference images of the SAME camera (T33). Each image is described by a compact global descriptor, a 32x18
gray thumbnail standardized to zero mean and unit variance (so day and night views of one scene stay close), and
the images are grouped with Ward hierarchical clustering. The number of groups is chosen by looking at the
dendrogram, the silhouette and the agreement with the file-name groups. Groups with fewer than 10 images are
discarded; each kept group sets aside 8 references spread over its time order, as `data.calibration_split` does.

Run:  python -m src.medellin_groups
Writes: data/splits/medellin_grupos.csv, results/figuras/medellin_dendrograma.png, results/medellin_k.csv.
A montage of sample groups for the visual check goes to data/ (not versioned: it shows pedestrians).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src import data

THUMB = (32, 18)  # width, height
MIN_GROUP = 10
N_REFERENCES = 8
CHOSEN_K = 80  # set after reading results/medellin_k.csv and the dendrogram (see docs/datos.md, section 7)
CANDIDATE_K = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]

IMAGES = data.ROOT / "data" / "raw" / "medellin" / "images"
INVENTORY = data.RESULTS_DIR / "inventario_medellin.csv"
GROUPS_CSV = data.SPLITS_DIR / "medellin_grupos.csv"
K_CSV = data.RESULTS_DIR / "medellin_k.csv"
DENDROGRAM = data.RESULTS_DIR / "figuras" / "medellin_dendrograma.png"
MONTAGE = data.ROOT / "data" / "medellin_grupos_muestra.png"


def thumbnail_descriptor(image: np.ndarray) -> np.ndarray:
    """Gray 32x18 thumbnail, standardized (zero mean, unit variance). A flat image gives zeros."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    spread = float(small.std())
    if spread < 1e-6:
        return np.zeros_like(small)
    return (small - small.mean()) / spread


def build_descriptors(files: list[str]) -> np.ndarray:
    rows = []
    for name in files:
        image = cv2.imread(str(IMAGES / name))
        if image is None:
            raise ValueError(f"cannot read image: {name}")
        rows.append(thumbnail_descriptor(image))
    return np.stack(rows)


def ward_tree(X: np.ndarray) -> np.ndarray:
    return linkage(X, method="ward")


def cut_tree(tree: np.ndarray, k: int) -> np.ndarray:
    """Cluster labels 1..k from cutting the dendrogram into k groups."""
    return fcluster(tree, t=k, criterion="maxclust")


def adjusted_rand(a, b) -> float:
    return float(adjusted_rand_score(a, b))


def evaluate_k(tree: np.ndarray, X: np.ndarray, ks: list[int], reference=None) -> pd.DataFrame:
    """For each candidate k: silhouette of the cut, and agreement (ARI) with the reference grouping if given."""
    rows = []
    for k in ks:
        labels = cut_tree(tree, k)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X, labels)) if 1 < len(set(labels)) < len(X) else np.nan,
                "ari_vs_nombre": adjusted_rand(reference, labels) if reference is not None else np.nan,
                "grupos_con_10_o_mas": int((pd.Series(labels).value_counts() >= MIN_GROUP).sum()),
            }
        )
    return pd.DataFrame(rows)


def assign_groups(inventory: pd.DataFrame, min_size: int = MIN_GROUP, n_refs: int = N_REFERENCES) -> pd.DataFrame:
    """Mark the groups kept (>= min_size images) and, inside each, n_refs references spread over time."""
    out = inventory.copy()
    sizes = out.groupby("cluster")["archivo"].transform("size")
    out["conservado"] = sizes >= min_size
    out["es_referencia"] = False
    for _, group in out[out["conservado"]].sort_values("instante").groupby("cluster"):
        positions = np.linspace(0, len(group) - 1, n_refs).round().astype(int)
        out.loc[group.index[np.unique(positions)], "es_referencia"] = True
    return out.sort_values(["cluster", "instante"]).reset_index(drop=True)


def load_inventory() -> pd.DataFrame:
    inv = pd.read_csv(INVENTORY, dtype={"fecha": str, "hora": str})
    inv["instante"] = inv["fecha"] + inv["hora"]
    return inv


def main() -> None:
    inventory = load_inventory()
    X = build_descriptors(inventory["archivo"].tolist())
    tree = ward_tree(X)
    names = inventory["grupo_nombre"].str.extract(r"(\d+)")[0].astype(int).to_numpy()

    table = evaluate_k(tree, X, CANDIDATE_K, reference=names)
    K_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(K_CSV, index=False)
    print(table.round(3).to_string(index=False))

    inventory["cluster"] = cut_tree(tree, CHOSEN_K)
    groups = assign_groups(inventory)
    data.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    groups[["archivo", "grupo_nombre", "cluster", "instante", "conservado", "es_referencia"]].to_csv(
        GROUPS_CSV, index=False
    )
    kept = groups[groups["conservado"]]
    print(
        f"k = {CHOSEN_K}: {kept['cluster'].nunique()} groups kept ({len(kept)} images, "
        f"{int(kept['es_referencia'].sum())} references), {int((~groups['conservado']).sum())} images discarded"
    )
    plot_dendrogram(tree)
    save_montage(groups)


def plot_dendrogram(tree: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    heights = tree[-160:, 2][::-1]  # the 160 largest merges, from the top of the tree down
    fig, (ax_tree, ax_elbow) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    dendrogram(tree, truncate_mode="lastp", p=CHOSEN_K, no_labels=True, ax=ax_tree)
    ax_tree.axhline(tree[-(CHOSEN_K - 1), 2], color="red", ls="--", lw=1, label=f"corte en k = {CHOSEN_K}")
    ax_tree.set(title="Dendrograma de Ward (últimas fusiones)", ylabel="distancia de fusión")
    ax_tree.legend()
    ax_elbow.plot(range(2, len(heights) + 2), heights, marker=".")
    ax_elbow.axvline(CHOSEN_K, color="red", ls="--", lw=1)
    ax_elbow.set(title="Altura de fusión según el número de grupos k", xlabel="k", ylabel="altura de la fusión")
    fig.tight_layout()
    DENDROGRAM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DENDROGRAM, dpi=130)
    plt.close(fig)


def save_montage(groups: pd.DataFrame, n_groups: int = 12, per_group: int = 6, seed: int = 0) -> None:
    """Thumbnails of random images of random kept groups, one row per group, for the manual check."""
    rng = np.random.default_rng(seed)
    kept = groups[groups["conservado"]]
    chosen = rng.choice(sorted(kept["cluster"].unique()), size=min(n_groups, kept["cluster"].nunique()), replace=False)
    rows = []
    for cluster in chosen:
        files = kept[kept["cluster"] == cluster]["archivo"].tolist()
        picks = rng.choice(files, size=min(per_group, len(files)), replace=False)
        tiles = [cv2.resize(cv2.imread(str(IMAGES / f)), (200, 112)) for f in picks]
        tiles += [np.zeros((112, 200, 3), np.uint8)] * (per_group - len(tiles))
        row = np.hstack(tiles)
        cv2.putText(row, f"grupo {cluster}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    MONTAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MONTAGE), np.vstack(rows))


if __name__ == "__main__":
    main()
