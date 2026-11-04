"""Train the MobileNetV2 classifier on the Bremen split (T62, T66).

One run = one set of hyperparameters. Each run writes results/cnn/<name>/ with history.csv (loss and accuracy per
epoch, train and val), curves.png, run.json (best val values and settings) and the weights (not versioned: *.keras).
Early stopping on the validation loss; the best weights are restored before saving.

Example: python -m src.train_cnn --name lr1e-3_b32 --lr 1e-3 --batch 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import cnn

RESULTS = Path(__file__).resolve().parents[1] / "results" / "cnn"


def run(
    name: str,
    learning_rate: float,
    batch: int,
    trainable_layers: int,
    epochs: int,
    seed: int = 0,
    augment: bool = False,
    dropout: float = 0.2,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    table = cnn.read_split()
    train = table[table["set"] == "train"]
    val = table[table["set"] == "val"]

    x_train = cnn.load_images(train["path"].tolist())
    y_train = train["y"].to_numpy()
    x_val = cnn.load_images(val["path"].tolist())
    y_val = val["y"].to_numpy()

    model = cnn.build_model(
        trainable_layers=trainable_layers, learning_rate=learning_rate, augment=augment, dropout=dropout
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    start = time.time()
    fit = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch,
        class_weight=cnn.class_weights(y_train),
        callbacks=[early],
        verbose=2,
    )
    seconds = time.time() - start

    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(fit.history)
    history.index.name = "epoch"
    history.to_csv(out / "history.csv")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set(title="loss", xlabel="epoch")
    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], label="val")
    ax_acc.set(title="accuracy", xlabel="epoch")
    ax_loss.legend()
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    best = int(np.argmin(history["val_loss"]))
    summary = {
        "name": name,
        "learning_rate": learning_rate,
        "batch": batch,
        "trainable_layers": trainable_layers,
        "augment": augment,
        "dropout": dropout,
        "epochs_run": len(history),
        "best_epoch": best + 1,
        "best_val_loss": float(history["val_loss"].iloc[best]),
        "best_val_accuracy": float(history["val_accuracy"].iloc[best]),
        "seconds": round(seconds, 1),
        "seed": seed,
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model.save(out / "model.keras")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--trainable", type=int, default=0, help="last base layers unfrozen (0 = frozen)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="random flip, translation, contrast, brightness")
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    summary = run(args.name, args.lr, args.batch, args.trainable, args.epochs, args.seed, args.augment, args.dropout)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
