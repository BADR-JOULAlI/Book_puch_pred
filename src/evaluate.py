from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate audiobook purchase classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model/best_model.keras"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/model/evaluation.json"))
    return parser.parse_args()


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["inputs"].astype(np.float32), data["targets"].astype(np.int32)


def main() -> None:
    args = parse_args()
    test_x, test_y = load_split(args.data_dir / "test.npz")
    model = tf.keras.models.load_model(args.model_path)

    probs = model.predict(test_x, verbose=0).reshape(-1)
    preds = (probs >= args.threshold).astype(np.int32)

    metrics = {
        "threshold": args.threshold,
        "roc_auc": float(roc_auc_score(test_y, probs)),
        "pr_auc": float(average_precision_score(test_y, probs)),
        "precision": float(precision_score(test_y, preds, zero_division=0)),
        "recall": float(recall_score(test_y, preds, zero_division=0)),
        "f1": float(f1_score(test_y, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(test_y, preds).tolist(),
        "classification_report": classification_report(test_y, preds, zero_division=0, output_dict=True),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print("Evaluation metrics:")
    print(
        f"ROC AUC={metrics['roc_auc']:.4f} | PR AUC={metrics['pr_auc']:.4f} | "
        f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}"
    )
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["confusion_matrix"])
    print(f"Saved evaluation report to: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
