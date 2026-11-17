"""Recover the camera of each Medellín image with unsupervised clustering (T17b, D15, class 20).

The file name (`imagenN_...`) is not a reliable camera id (55 groups, 22 with 1 to 4 images), but the classic module
needs reference images of the SAME camera (T33). Each image is described by a compact global descriptor, a 32x18
gray thumbnail standardized to zero mean and unit variance (so day and night views of one scene stay close), and
the images are grouped with Ward hierarchical clustering. The number of groups is chosen by looking at the
dendrogram, the silhouette and the agreement with the file-name groups. Groups with fewer than 10 images are
discarded; each kept group sets aside 8 references spread over its time order, as `data.calibration_split` does.

Run:  python -m src.medellin_groups            (everything, about 15 minutes)
      python -m src.medellin_groups --control  (only the negative control, about 5 minutes)
Writes: data/splits/medellin_grupos.csv, results/figuras/medellin_dendrograma.png, results/medellin_k.csv (with
the ORB match rate of each k in ORB_KS) and results/medellin_control_negativo.csv. It takes about 15 minutes.
A montage of sample groups for the visual check goes to data/ (not versioned: it shows pedestrians).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src import data, features

THUMB = (32, 18)  # width, height
MIN_GROUP = 10
N_REFERENCES = 8
MIN_MATCH_RATE = 0.7  # below this share of ORB matches with its own references a group is flagged as doubtful
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


def evaluate_k(tree: np.ndarray, X: np.ndarray, ks: list[int], reference=None, extra=None) -> pd.DataFrame:
    """For each candidate k: silhouette of the cut, and agreement (ARI) with the reference grouping if given.

    extra(k, labels) may return more columns for that k (for example its ORB match rate); it is the way to save in
    the same table every number that is used to choose k.
    """
    rows = []
    for k in ks:
        labels = cut_tree(tree, k)
        row = {
            "k": k,
            "silhouette": float(silhouette_score(X, labels)) if 1 < len(set(labels)) < len(X) else np.nan,
            "ari_vs_nombre": adjusted_rand(reference, labels) if reference is not None else np.nan,
            "grupos_con_10_o_mas": int((pd.Series(labels).value_counts() >= MIN_GROUP).sum()),
        }
        if extra is not None:
            row.update(extra(k, labels))
        rows.append(row)
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


def match_rate_per_group(
    groups: pd.DataFrame, load_gray, matcher_cls=features.ReferenceMatcher, sample: int = 20, match_min: float = 12.5
) -> pd.DataFrame:
    """Share of each kept group's images that match the group's own references (what the classic module needs).

    `load_gray(name)` returns the 640x360 gray image. At most `sample` non-reference images per group, evenly
    spaced in time, are matched against the 8 references; a match counts when orb_matches >= match_min.
    A group that mixes cameras shows a low rate because its references belong to only one of them.
    """
    rows = {}
    for cluster, group in groups[groups["conservado"]].sort_values("instante").groupby("cluster"):
        refs = group[group["es_referencia"]]["archivo"]
        rest = group[~group["es_referencia"]]["archivo"].tolist()
        if len(rest) > sample:
            rest = [rest[i] for i in np.linspace(0, len(rest) - 1, sample).round().astype(int)]
        matcher = matcher_cls([load_gray(name) for name in refs])
        hits = sum(matcher.match(load_gray(name))["orb_matches"] >= match_min for name in rest)
        rows[cluster] = {"evaluadas": len(rest), "tasa": hits / len(rest) if rest else float("nan")}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("cluster")


def nearest_clusters(X: np.ndarray, labels: np.ndarray, k: int = 5) -> dict[int, list[int]]:
    """For each cluster, the k other clusters whose centroids (mean thumbnail descriptor) are closest."""
    ids = sorted(set(labels))
    centroids = np.stack([X[labels == c].mean(axis=0) for c in ids])
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    out = {}
    for i, c in enumerate(ids):
        order = [ids[j] for j in np.argsort(dist[i]) if ids[j] != c]
        out[c] = [int(o) for o in order[:k]]
    return out


def control_negativo(
    groups: pd.DataFrame, load_gray, near: dict[int, list[int]], random: dict[int, list[int]],
    matcher_cls=features.ReferenceMatcher, sample: int = 10, match_min: float = 12.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Negative control of match_rate_per_group: the same images against the references of OTHER groups.

    A high own rate only means something if the rate against other groups is low. For each kept group, `sample`
    non-reference images (evenly spaced in time) are matched against the references of its own group, of the
    clusters listed in `near[cluster]` (the closest ones, where the same camera split in two shows up) and of those
    in `random[cluster]` (far away, which measure how often ORB matches a different scene by chance).
    Returns a summary per group and the table of every (group, other group) pair with its match rate.
    """
    kept = groups[groups["conservado"]].sort_values("instante")
    available = set(kept["cluster"])
    matchers: dict[int, object] = {}

    def matcher_of(cluster: int):
        if cluster not in matchers:
            refs = kept[(kept["cluster"] == cluster) & kept["es_referencia"]]["archivo"]
            matchers[cluster] = matcher_cls([load_gray(name) for name in refs])
        return matchers[cluster]

    summary, pair_rows = {}, []
    for cluster, group in kept.groupby("cluster"):
        rest = group[~group["es_referencia"]]["archivo"].tolist()
        if len(rest) > sample:
            rest = [rest[i] for i in np.linspace(0, len(rest) - 1, sample).round().astype(int)]
        kinds = {c: "cercano" for c in near.get(cluster, []) if c in available and c != cluster}
        kinds.update({c: "azar" for c in random.get(cluster, []) if c in available and c != cluster and c not in kinds})
        hits = {c: 0 for c in [cluster] + list(kinds)}
        for name in rest:
            gray = load_gray(name)
            for c in hits:
                hits[c] += matcher_of(c).match(gray)["orb_matches"] >= match_min
        n = len(rest)
        pair_rows += [{"cluster": cluster, "otro": c, "tipo": kind, "tasa": hits[c] / n} for c, kind in kinds.items() if n]
        near_rates = {c: hits[c] / n for c, k in kinds.items() if k == "cercano" and n}
        far_rates = [hits[c] / n for c, k in kinds.items() if k == "azar" and n]
        best = max(near_rates, key=near_rates.get) if near_rates else -1
        summary[cluster] = {
            "evaluadas": n,
            "tasa_propia": hits[cluster] / n if n else float("nan"),
            "tasa_cercana_media": float(np.mean(list(near_rates.values()))) if near_rates else float("nan"),
            "tasa_cercana_max": near_rates[best] if near_rates else float("nan"),
            "grupo_cercano_max": best,
            "tasa_azar_media": float(np.mean(far_rates)) if far_rates else float("nan"),
        }
    pairs = pd.DataFrame(pair_rows, columns=["cluster", "otro", "tipo", "tasa"])
    return pd.DataFrame.from_dict(summary, orient="index").rename_axis("cluster"), pairs


def _load_gray(name: str) -> np.ndarray:
    return features.preprocess(features.load_image(IMAGES / name))["gray"]


def load_inventory() -> pd.DataFrame:
    inv = pd.read_csv(INVENTORY, dtype={"fecha": str, "hora": str})
    inv["instante"] = inv["fecha"] + inv["hora"]
    return inv


ORB_KS = [40, 60, 80, 100, 120, 160]  # the k values whose ORB match rate is measured (it takes about a minute each)
CONTROL_NEIGHBORS = 5  # nearest other groups used as hard negatives
CONTROL_RANDOM = 3  # random other groups, to also test far away ones
CONTROL_CSV = data.RESULTS_DIR / "medellin_control_negativo.csv"
CONTROL_PAIRS_CSV = data.RESULTS_DIR / "medellin_control_pares.csv"


def run_control(groups: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> None:
    """Negative control at the chosen k: 5 nearest groups and 3 random far ones per group; saves summary and pairs."""
    near = nearest_clusters(X, labels, CONTROL_NEIGHBORS)
    rng = np.random.default_rng(0)
    ids = sorted(set(labels))
    random = {
        c: [int(o) for o in rng.choice([i for i in ids if i != c and i not in near[c]], CONTROL_RANDOM, replace=False)]
        for c in ids
    }
    summary, pairs = control_negativo(groups, _load_gray, near, random)
    summary.to_csv(CONTROL_CSV)
    pairs.to_csv(CONTROL_PAIRS_CSV, index=False)
    weights = summary["evaluadas"]
    print(f"negative control, {len(summary)} groups:")
    print(f"  own rate (weighted): {(summary.tasa_propia * weights).sum() / weights.sum():.3f}")
    for kind in ("cercano", "azar"):
        part = pairs[pairs["tipo"] == kind]
        print(
            f"  against {kind} groups: mean {part['tasa'].mean():.3f}, {(part['tasa'] >= 0.5).mean():.1%} of "
            f"{len(part)} pairs at 50 % or more"
        )


def main_control() -> None:
    """Only the negative control, from the groups already saved in data/splits/medellin_grupos.csv."""
    inventory = load_inventory()
    X = build_descriptors(inventory["archivo"].tolist())
    groups = pd.read_csv(GROUPS_CSV, dtype={"instante": str})
    labels = inventory["archivo"].map(groups.set_index("archivo")["cluster"]).to_numpy()
    run_control(groups, X, labels)


def main() -> None:
    inventory = load_inventory()
    X = build_descriptors(inventory["archivo"].tolist())
    tree = ward_tree(X)
    names = inventory["grupo_nombre"].str.extract(r"(\d+)")[0].astype(int).to_numpy()

    def orb_rate(k: int, labels: np.ndarray) -> dict:
        if k not in ORB_KS:
            return {"tasa_orb_propia": np.nan}
        grouped = assign_groups(inventory.assign(cluster=labels))
        rates = match_rate_per_group(grouped, _load_gray)
        return {"tasa_orb_propia": float((rates["tasa"] * rates["evaluadas"]).sum() / rates["evaluadas"].sum())}

    table = evaluate_k(tree, X, CANDIDATE_K, reference=names, extra=orb_rate)
    K_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(K_CSV, index=False)
    print(table.round(3).to_string(index=False))

    labels = cut_tree(tree, CHOSEN_K)
    inventory["cluster"] = labels
    groups = assign_groups(inventory)
    rates = match_rate_per_group(groups, _load_gray)
    groups["tasa_orb_grupo"] = groups["cluster"].map(rates["tasa"])
    groups["confiable"] = groups["conservado"] & (groups["tasa_orb_grupo"] >= MIN_MATCH_RATE)
    print(f"ORB match rate with the group's own references: {(rates.tasa * rates.evaluadas).sum() / rates.evaluadas.sum():.3f}")
    print(f"doubtful groups (< {MIN_MATCH_RATE:.0%}):", rates[rates["tasa"] < MIN_MATCH_RATE].round(2).to_dict("index"))
    data.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    groups[["archivo", "grupo_nombre", "cluster", "instante", "conservado", "es_referencia", "tasa_orb_grupo", "confiable"]].to_csv(
        GROUPS_CSV, index=False
    )
    kept = groups[groups["conservado"]]
    print(
        f"k = {CHOSEN_K}: {kept['cluster'].nunique()} groups kept ({len(kept)} images, "
        f"{int(kept['es_referencia'].sum())} references), {int((~groups['conservado']).sum())} images discarded"
    )

    run_control(groups, X, labels)
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
    import sys

    main_control() if "--control" in sys.argv[1:] else main()
