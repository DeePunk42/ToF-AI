"""
Train a hand-gesture classifier on the ST VL53L8CX dataset using Keras.

- Loads NPZ files from: model/datasets/ST_VL53L8CX_handposture_dataset
- Preprocesses distance_mm and signal_per_spad channels per SUMMARY_CN.md
- Uses each time frame from every NPZ as an independent sample (no temporal compression)
- Splits data in a stratified way into train/val/test
- Trains a lightweight ConvNet compatible with the existing deployment shape (8x8x2)
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf


CLASS_NAMES = [
    "BreakTime",
    "CrossHands",
    "Dislike",
    "Fist",
    "FlatHand",
    "Like",
    "Love",
    "None",
]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_frames(npz_path: pathlib.Path) -> np.ndarray:
    """Load a single NPZ and return all time frames as (T, 8, 8, 2) float32."""
    data = np.load(npz_path)
    zone = data["zone_data"]  # shape (4, 64, T)

    # Channels: 0=target_status, 1=valid, 2=signal_per_spad, 3=distance_mm
    signal = zone[2]  # (64, T)
    distance = zone[3]  # (64, T)
    time_len = signal.shape[1]

    frames: List[np.ndarray] = []
    for t in range(time_len):
        signal_frame = signal[:, t]
        distance_frame = distance[:, t]

        # Normalization per SUMMARY_CN.md
        signal_norm = np.clip(signal_frame / 5000.0, 0.0, 1.0)
        distance_norm = np.clip((distance_frame - 100.0) / 300.0, 0.0, 1.0)

        signal_img = signal_norm.reshape(8, 8)
        distance_img = distance_norm.reshape(8, 8)
        stacked = np.stack([distance_img, signal_img], axis=-1).astype(np.float32)
        frames.append(stacked)

    return np.stack(frames, axis=0)


def collect_dataset(root: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all samples and labels into arrays; each time frame becomes a sample."""
    samples: List[np.ndarray] = []
    labels: List[int] = []

    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for npz_path in class_dir.rglob("*.npz"):
            frames = load_frames(npz_path)  # (T, 8, 8, 2)
            samples.extend(list(frames))
            labels.extend([LABEL_TO_INDEX[class_name]] * frames.shape[0])

    x = np.stack(samples, axis=0)
    y = np.array(labels, dtype=np.int64)
    return x, y


def stratified_split(
    labels: Iterable[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices with simple per-class stratification."""
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in range(len(CLASS_NAMES)):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        total = len(cls_indices)
        n_test = int(round(total * test_ratio))
        n_val = int(round(total * val_ratio))
        # Ensure we do not exhaust the class
        if n_test + n_val >= total:
            n_test = min(n_test, max(total - 1, 0))
            n_val = min(n_val, max(total - n_test - 1, 0))
        val_idx.extend(cls_indices[:n_val])
        test_idx.extend(cls_indices[n_val : n_val + n_test])
        train_idx.extend(cls_indices[n_val + n_test :])

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def make_class_weights(labels: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total = len(labels)
    weights = {i: total / (len(CLASS_NAMES) * count) for i, count in enumerate(counts) if count > 0}
    return weights


def build_model(learning_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(8, 8, 2))
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_training(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_root = pathlib.Path(args.data_root)
    x, y = collect_dataset(data_root)
    print(f"Loaded {len(x)} samples from {data_root}")
    for idx, name in enumerate(CLASS_NAMES):
        count = int((y == idx).sum())
        print(f"  {name}: {count}")

    train_idx, val_idx, test_idx = stratified_split(y, args.val_ratio, args.test_ratio, args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=len(CLASS_NAMES))

    class_weights = make_class_weights(y_train)
    print("Class weights:", class_weights)
    print(
        f"Splits -> train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)} "
        f"(val_ratio={args.val_ratio}, test_ratio={args.test_ratio})"
    )

    model = build_model(args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    if args.history:
        np.savez(args.history, **{k: np.array(v) for k, v in history.history.items()})
        print(f"Saved training history to {args.history}")

    # Ensure final model is saved (ModelCheckpoint saves best already)
    model.save(args.output)
    print(f"Saved model to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand-gesture model on VL53L8CX dataset.")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("model/datasets/ST_VL53L8CX_handposture_dataset"),
        help="Root folder containing class subdirectories with NPZ files.",
    )
    parser.add_argument("--output", type=str, default="handpose_best.h5", help="Path to save the trained model.")
    parser.add_argument("--history", type=str, default="", help="Optional path to save training history as NPZ.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio per class.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio per class.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
