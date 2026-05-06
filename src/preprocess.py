from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess audiobooks dataset.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Audiobooks_data.csv"),
        help="Path to raw CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
        help="Output directory for split datasets and scaler.",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Validation ratio over the full dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.loadtxt(args.input_csv, delimiter=",")
    inputs = raw[:, 1:-1]
    targets = raw[:, -1].astype(np.int32)

    val_ratio_adjusted = args.validation_size / (1.0 - args.test_size)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        inputs,
        targets,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=targets,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=args.seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_val).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    np.savez(args.output_dir / "train.npz", inputs=x_train_scaled, targets=y_train)
    np.savez(args.output_dir / "validation.npz", inputs=x_val_scaled, targets=y_val)
    np.savez(args.output_dir / "test.npz", inputs=x_test_scaled, targets=y_test)
    joblib.dump(scaler, args.output_dir / "scaler.joblib")

    pos_rate = float(np.mean(targets))
    print(f"Raw samples: {len(targets)}, positive rate: {pos_rate:.4f}")
    print(f"Train: {x_train_scaled.shape}, positives: {int(y_train.sum())}/{len(y_train)}")
    print(f"Val:   {x_val_scaled.shape}, positives: {int(y_val.sum())}/{len(y_val)}")
    print(f"Test:  {x_test_scaled.shape}, positives: {int(y_test.sum())}/{len(y_test)}")
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
