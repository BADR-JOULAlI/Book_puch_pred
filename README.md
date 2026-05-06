# Book Purchase Prediction

Production-ready binary classification pipeline to predict whether a customer will purchase an audiobook again.

## Overview

This project contains a complete machine learning workflow:
- Data preprocessing with strict train/validation/test separation
- Neural network training with TensorFlow/Keras
- Class-imbalance handling using `class_weight`
- Evaluation with business-relevant metrics (ROC AUC, PR AUC, Precision, Recall, F1)
- Reproducible artifacts for model, scaler, history, and reports

## Key Features

- No data leakage: scaler is fit on training data only
- Stratified splitting to preserve class distribution
- Reproducible runs via fixed random seed
- Early stopping and best-model checkpointing
- Script-first pipeline + equivalent notebooks

## Project Structure

```text
.
├── Audiobooks_data.csv
├── src
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── artifacts
│   ├── data
│   │   ├── train.npz
│   │   ├── validation.npz
│   │   ├── test.npz
│   │   └── scaler.joblib
│   └── model
│       ├── best_model.keras
│       ├── history.json
│       ├── train_config.json
│       └── evaluation.json
├── TensorFlow Audiobooks - Preprocessing with Comments.ipynb
├── prediction.ipynb
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- pip

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

1. Preprocess data

```bash
python src/preprocess.py --input-csv Audiobooks_data.csv
```

2. Train model

```bash
python src/train.py
```

3. Evaluate model

```bash
python src/evaluate.py
```

Final evaluation report:

- `artifacts/model/evaluation.json`

## Notebooks

Both notebooks are aligned with the script pipeline:
- `TensorFlow Audiobooks - Preprocessing with Comments.ipynb`
- `prediction.ipynb`

Use scripts for reproducible runs and notebooks for interactive analysis.

## Example Metrics

Recent run (threshold = 0.5):
- ROC AUC: `0.9009`
- PR AUC: `0.7510`
- Precision: `0.5625`
- Recall: `0.6830`
- F1-score: `0.6169`

## Current Scope

This repository currently focuses on:
- Data preprocessing
- Model training
- Model evaluation
- Saving reproducible artifacts

## License

Add your preferred license file (`LICENSE`) before publishing publicly.
