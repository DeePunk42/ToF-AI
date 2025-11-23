"""
Quantize a trained Keras gesture model to TFLite (INT8) and evaluate accuracy.

Usage (from repo root):
  .venv/bin/python quantize_tflite.py --model handpose_best.h5 --output handpose_int8.tflite
"""

from __future__ import annotations

import argparse
import pathlib
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


def representative_data_gen(x_train: np.ndarray, batch_size: int = 1):
    """Yield samples for INT8 calibration."""
    for i in range(0, len(x_train), batch_size):
        batch = x_train[i : i + batch_size]
        yield [batch.astype(np.float32)]


def evaluate_tflite(interpreter: tf.lite.Interpreter, x: np.ndarray, y: np.ndarray) -> float:
    """Compute accuracy of a TFLite model on given data."""
    input_index = interpreter.get_input_details()[0]["index"]
    input_scale, input_zero_point = interpreter.get_input_details()[0]["quantization"]
    output_details = interpreter.get_output_details()[0]
    output_index = output_details["index"]
    out_scale, out_zero = output_details["quantization"]

    correct = 0
    total = len(x)

    for sample, label in zip(x, y):
        q_input = np.round(sample / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_index, q_input[np.newaxis, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0]
        scores = (output.astype(np.float32) - out_zero) * out_scale
        pred = int(np.argmax(scores))
        correct += int(pred == label)

    return correct / total if total else 0.0


def main(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_root = pathlib.Path(args.data_root)
    x, y = collect_dataset(data_root)
    print(f"Loaded frames: {len(x)} from {data_root}")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == idx).sum()} frames")

    train_idx, val_idx, test_idx = stratified_split(y, args.val_ratio, args.test_ratio, args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(
        f"Splits (frames) -> train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)} "
        f"(val_ratio={args.val_ratio}, test_ratio={args.test_ratio})"
    )

    # Load Keras model and check baseline accuracy
    model = tf.keras.models.load_model(args.model)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=len(CLASS_NAMES))
    _, keras_acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"Keras test accuracy: {keras_acc:.4f}")

    # Build converter with INT8 full quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(x_train, batch_size=1)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path = pathlib.Path(args.output)
    output_path.write_bytes(tflite_model)
    print(f"Saved TFLite INT8 model to {output_path}")

    # Evaluate TFLite accuracy
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    tflite_acc = evaluate_tflite(interpreter, x_test, y_test)
    print(f"TFLite INT8 test accuracy: {tflite_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a Keras hand-gesture model to TFLite INT8.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Keras model (.h5 or .keras).")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("model/datasets/ST_VL53L8CX_handposture_dataset"),
        help="Root folder containing class subdirectories with NPZ files.",
    )
    parser.add_argument("--output", type=str, default="handpose_int8.tflite", help="Output path for TFLite model.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio per class.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
