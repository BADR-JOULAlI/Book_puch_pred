from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train audiobook purchase classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["inputs"].astype(np.float32), data["targets"].astype(np.int32)


def build_model(input_size: int, hidden_size: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split(args.data_dir / "train.npz")
    val_x, val_y = load_split(args.data_dir / "validation.npz")

    pos = int(train_y.sum())
    neg = len(train_y) - pos
    class_weight = {0: len(train_y) / (2.0 * max(neg, 1)), 1: len(train_y) / (2.0 * max(pos, 1))}

    model = build_model(train_x.shape[1], args.hidden_size, args.learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=args.patience, mode="max", restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(args.model_dir / "best_model.keras"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    )

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[early_stopping, checkpoint],
        verbose=2,
    )

    with (args.model_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history.history, fp, indent=2)

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "class_weight": class_weight,
    }
    with (args.model_dir / "train_config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)

    print(f"Training complete. Best model saved at: {(args.model_dir / 'best_model.keras').resolve()}")


if __name__ == "__main__":
    main()
