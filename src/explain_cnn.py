"""Grad-CAM explanations of the CNN: where does it look? (T67)

Grad-CAM weights the 7x7 feature maps of the last convolution of MobileNetV2 (`out_relu`, 1280 channels) by the
mean gradient of the class score (the logit before the softmax) and keeps the positive part. The result is a coarse
heat map of the image zones that push the model toward that class. It answers: does the model look at the sabotage
(the cover, the blur, the changed scene) or at the context (light, sky, vegetation)?

Outputs: results/figuras/gradcam_*.png and results/cnn/<run>/gradcam_resumen.csv. Run:
    python -m src.explain_cnn
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from src import cnn, data

ROOT = data.ROOT
DEFAULT_RUN = "lr1e-2_b32_frozen"
LAST_CONV = "out_relu"
THUMB = (240, 135)


def base_of(model):
    """The nested MobileNetV2 of the model (the layer that contains `out_relu`)."""
    for layer in model.layers:
        if hasattr(layer, "get_layer"):
            try:
                layer.get_layer(LAST_CONV)
                return layer
            except ValueError:
                continue
    raise ValueError("the model has no MobileNetV2 base with an out_relu layer")


def grad_cam(model, images: np.ndarray, class_index=None, batch_size: int = 32) -> dict:
    """Heat maps (n, 7, 7) scaled to [0, 1] per image, plus predicted classes and probabilities.

    class_index: None explains the predicted class; an int or a sequence explains that class for every image.
    """
    import tensorflow as tf

    base = base_of(model)
    extractor = tf.keras.Model(base.input, [base.get_layer(LAST_CONV).output, base.output])
    kernel, bias = (tf.constant(w) for w in model.layers[-1].get_weights())
    heatmaps, preds, probas, explained = [], [], [], []
    for start in range(0, len(images), batch_size):
        batch = tf.constant(images[start:start + batch_size])
        with tf.GradientTape() as tape:
            conv, pooled = extractor(batch, training=False)
            tape.watch(conv)
            logits = tf.matmul(pooled, kernel) + bias
            predicted = tf.argmax(logits, axis=1).numpy()
            if class_index is None:
                chosen = predicted
            else:
                chosen = np.broadcast_to(np.asarray(class_index), (len(images),))[start:start + batch_size]
            score = tf.gather(logits, tf.constant(chosen, dtype=tf.int32), batch_dims=1)
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1)).numpy()
        peak = cam.max(axis=(1, 2), keepdims=True)
        heatmaps.append(np.divide(cam, peak, out=np.zeros_like(cam), where=peak > 0))
        preds.append(predicted)
        probas.append(tf.nn.softmax(logits).numpy())
        explained.append(np.asarray(chosen))
    return {
        "heatmaps": np.concatenate(heatmaps),
        "pred": np.concatenate(preds),
        "proba": np.concatenate(probas),
        "explained": np.concatenate(explained),
    }


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a JET-colored heat map over a BGR image; zones with zero heat stay untouched."""
    height, width = image.shape[:2]
    heat = np.clip(cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    weight = (alpha * heat)[..., None]
    return np.clip(image.astype(np.float32) * (1 - weight) + color * weight, 0, 255).astype(np.uint8)


def _third_weights(rows: int) -> np.ndarray:
    """(3, rows) matrix: the part of each row that falls in each vertical third of the image.

    Rows do not split into whole thirds (7 rows are 2, 3 and 2 by their centers), so a row that straddles a
    boundary is shared in proportion to the area on each side. A uniform map then gives exactly 1/3 per third.
    """
    edges = np.arange(rows + 1) / rows
    weights = np.zeros((3, rows))
    for third in range(3):
        low, high = third / 3, (third + 1) / 3
        weights[third] = np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0, None) * rows
    return weights


def attention_stats(heatmaps: np.ndarray) -> pd.DataFrame:
    """Where the attention mass sits: centroid, share per vertical third (a uniform map gives 1/3 each) and
    concentration (0 uniform, 1 one cell)."""
    n, rows, cols = heatmaps.shape
    ys = (np.arange(rows) + 0.5) / rows
    xs = (np.arange(cols) + 0.5) / cols
    thirds = _third_weights(rows)
    out = []
    for heat in heatmaps:
        mass = float(heat.sum())
        if mass <= 0:
            out.append({"masa": 0.0, "centro_x": 0.5, "centro_y": 0.5, "tercio_superior": 0.0, "tercio_medio": 0.0,
                        "tercio_inferior": 0.0, "concentracion": 0.0})
            continue
        p = heat / mass
        row_share = p.sum(axis=1)
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        out.append(
            {
                "masa": mass,
                "centro_x": float((p.sum(axis=0) * xs).sum()),
                "centro_y": float((row_share * ys).sum()),
                "tercio_superior": float(thirds[0] @ row_share),
                "tercio_medio": float(thirds[1] @ row_share),
                "tercio_inferior": float(thirds[2] @ row_share),
                "concentracion": 1.0 - entropy / np.log(rows * cols),
            }
        )
    return pd.DataFrame(out)


def pick_examples(table: pd.DataFrame, per_class: int = 4, seed: int = 0) -> pd.DataFrame:
    """per_class random rows of each class (all of them when the class has fewer), reproducible."""
    parts = [
        group.sample(min(per_class, len(group)), random_state=seed)
        for _, group in table.groupby("label", sort=False)
    ]
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------- figures and run


def explain_table(model, table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Grad-CAM of the predicted class for every image: a table (prediction, attention stats) and the heat maps."""
    images = cnn.load_images(table["path"].tolist())
    result = grad_cam(model, images)
    out = table[["path", "label"]].reset_index(drop=True).copy()
    out["pred"] = [cnn.CLASSES[i] for i in result["pred"]]
    out["prob"] = result["proba"].max(axis=1)
    out = pd.concat([out, attention_stats(result["heatmaps"])], axis=1)
    return out, result["heatmaps"]


def plot_grid(rows: list[tuple[str, pd.DataFrame, np.ndarray]], per_row: int, title: str):
    """One row per (label, examples, heatmaps): the image with its heat map, titled real -> predicted (probability)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), per_row, figsize=(3.2 * per_row, 1.9 * len(rows) + 0.6), squeeze=False)
    for r, (name, examples, heat) in enumerate(rows):
        for c in range(per_row):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(examples):
                continue
            row = examples.iloc[c]
            image = cv2.resize(cv2.imread(str(cnn.BREMEN_ROOT / row["path"])), THUMB, interpolation=cv2.INTER_AREA)
            ax.imshow(cv2.cvtColor(overlay(image, heat[c]), cv2.COLOR_BGR2RGB))
            ok = row["label"] == row["pred"]
            ax.set_title(f"{row['label']} → {row['pred']} ({row['prob']:.2f})", fontsize=8,
                         color="black" if ok else "firebrick")
        axes[r, 0].text(-0.05, 0.5, name, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()
    run_dir = ROOT / "results" / "cnn" / args.run
    figures = ROOT / "results" / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(run_dir / "model.keras")

    split = cnn.read_split()
    sets = {"test": split[split["set"] == "test"].reset_index(drop=True)}
    unseen = pd.read_csv(ROOT / "data" / "splits" / "camara_no_vista.csv")
    refs = pd.read_csv(ROOT / "data" / "splits" / "referencias_camara_no_vista.csv")
    sets["camara_no_vista"] = unseen[~unseen["path"].isin(refs["path"])].reset_index(drop=True)

    summaries = []
    for name, table in sets.items():
        explained, heat = explain_table(model, table)
        explained["conjunto"] = name
        explained["acierto"] = explained["label"] == explained["pred"]
        summaries.append(explained)

        picks = pick_examples(explained, args.per_class)
        rows = []
        for label in cnn.CLASSES:
            part = picks[picks["label"] == label]
            if len(part):
                rows.append((label, part, heat[[explained.index[explained["path"] == p][0] for p in part["path"]]]))
        plot_grid(rows, args.per_class, f"Grad-CAM, {name} (rojo: error)").savefig(
            figures / f"gradcam_{name}.png", dpi=110
        )

    everything = pd.concat(summaries, ignore_index=True)
    stats = (
        everything.groupby(["conjunto", "label", "acierto"])[
            ["centro_x", "centro_y", "tercio_superior", "tercio_medio", "tercio_inferior", "concentracion", "prob"]
        ]
        .mean()
        .round(3)
        .join(everything.groupby(["conjunto", "label", "acierto"]).size().rename("n"))
        .reset_index()
    )
    stats.to_csv(run_dir / "gradcam_resumen.csv", index=False)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
