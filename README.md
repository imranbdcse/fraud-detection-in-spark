# Real-Time Credit Card Fraud Detection with Apache Spark

A production-ready pipeline for detecting credit card fraud in real time using
Apache Spark Structured Streaming, Apache Kafka, and Scikit-learn.

## Project Structure

```
fraud-detection-in-spark/
├── data/
│   └── generate_synthetic_data.py   # Synthetic dataset generator
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── README.md
├── src/
│   ├── batch/
│   │   ├── train_model.py           # Training pipeline
│   │   ├── predict.py               # Inference helper
│   │   └── batch_processor.py       # Batch processing
│   └── streaming/
│       ├── fraud_detector.py        # Basic Spark streaming job
│       ├── fraud_detector_enhanced.py # Enhanced with feature engineering
│       └── README.md
├── models/                          # Saved model artefacts (generated)
├── tests/
│   └── test_end_to_end.py           # Integration tests
├── scripts/
│   ├── kafka_producer.py            # Kafka transaction producer
│   ├── start_streaming.sh
│   └── stop_streaming.sh
├── env/
│   ├── config.env                   # Environment variables
│   └── spark_config.conf            # Spark configuration
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.7+
- Apache Spark 2.4+ (with Kafka connector)
- Apache Kafka 2.8+
- Java 8+

## Installation

```bash
# Clone the repository
git clone https://github.com/imranbdcse/fraud-detection-in-spark.git
cd fraud-detection-in-spark

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Data

```bash
python data/generate_synthetic_data.py
```

Creates `data/creditcard.csv` with 100 000 transactions (2 % fraud rate).

### 2. Train the Model

```bash
python src/batch/train_model.py
# With custom paths:
python src/batch/train_model.py \
    --data-path data/creditcard.csv \
    --model-dir models/ \
    --test-size 0.2
```

Saves `models/fraud_model.pkl` and `models/scaler.pkl`.

### 3. Run Batch Predictions

```bash
python src/batch/predict.py --input data/creditcard.csv --output predictions.csv
```

### 4. Run Batch Processing

```bash
python src/batch/batch_processor.py \
    --input data/creditcard.csv \
    --format csv \
    --output-dir batch_results/
```

### 5. Start Real-Time Streaming

```bash
# Source environment variables
source env/config.env

# Start all components (Kafka producer + Spark streaming)
chmod +x scripts/start_streaming.sh scripts/stop_streaming.sh
./scripts/start_streaming.sh

# Or submit Spark job directly
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    --properties-file env/spark_config.conf \
    src/streaming/fraud_detector.py

# Stop all components
./scripts/stop_streaming.sh
```

### 6. Run Notebooks

```bash
jupyter notebook notebooks/
```

Open `01_exploratory_data_analysis.ipynb` then `02_model_training.ipynb`.

## Running Tests

```bash
pytest tests/test_end_to_end.py -v
```

## Configuration

| File | Purpose |
|------|---------|
| `env/config.env` | Kafka, model, and path environment variables |
| `env/spark_config.conf` | Spark memory, executor, and streaming settings |

Key environment variables (see `env/config.env` for full list):

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_INPUT_TOPIC=transactions
KAFKA_OUTPUT_TOPIC=fraud_alerts
MODEL_PATH=models/fraud_model.pkl
SCALER_PATH=models/scaler.pkl
```

## Data Flow

```
Kafka Producer
      │
      ▼
Kafka topic: transactions
      │
      ▼
Spark Structured Streaming
  ├── Parse JSON
  ├── Feature engineering (enhanced)
  ├── Apply ML model (Pandas UDF)
  └── Write to sinks:
        ├── Console (monitoring)
        ├── Parquet (storage)
        └── Kafka topic: fraud_alerts
```

## Model Details

- **Algorithm**: Logistic Regression with `class_weight='balanced'`
- **Scaler**: StandardScaler
- **Evaluation**: ROC-AUC, F1-Score, Precision, Recall, cross-validation
- **Serialisation**: joblib

---

## Code Overview

Below is a detailed walkthrough of every source file in the project, showing the
full code and describing what each component does.

---

### `data/generate_synthetic_data.py` — Synthetic Dataset Generator

Generates a realistic imbalanced credit card transaction dataset with 100 000
samples and a 2 % fraud ratio, matching the format of the Kaggle Credit Card
Fraud Detection dataset. It produces 31 columns: `Time`, 28 PCA-like anonymous
features `V1`–`V28`, `Amount`, and the binary target `Class` (0 = normal,
1 = fraud). Fraudulent transactions have shifted feature means and higher
variance amounts to simulate real-world statistical differences.

```python
"""
Generate synthetic credit card transaction dataset for fraud detection.

Creates a realistic imbalanced dataset with 100,000 samples and 2% fraud ratio,
matching the format of the Kaggle Credit Card Fraud Detection dataset.
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_data(
    n_samples: int = 100000,
    fraud_ratio: float = 0.02,
    random_state: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Fraction of transactions that are fraudulent.
        random_state: Random seed for reproducibility.
        output_path: Path to save the CSV file. Defaults to data/creditcard.csv.

    Returns:
        DataFrame containing the synthetic dataset.
    """
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    print(f"Generating {n_samples:,} transactions ({n_normal:,} normal, {n_fraud:,} fraudulent)...")

    # Time feature (seconds elapsed over ~2 days)
    time_normal = np.sort(rng.uniform(0, 172800, n_normal))
    time_fraud = rng.uniform(0, 172800, n_fraud)
    time_all = np.concatenate([time_normal, time_fraud])

    # V1-V28: PCA-like anonymous features
    # Normal transactions cluster around zero; fraud has shifted means.
    normal_features = rng.randn(n_normal, 28)

    # Fraud transactions have different statistical properties
    fraud_shift = rng.uniform(-3, 3, 28)
    fraud_features = rng.randn(n_fraud, 28) * 1.5 + fraud_shift

    features_all = np.vstack([normal_features, fraud_features])

    # Amount feature
    # Normal: log-normal with lower amounts; Fraud: slightly higher variance
    amount_normal = np.exp(rng.normal(3.5, 1.5, n_normal)).clip(0.5, 5000)
    amount_fraud = np.exp(rng.normal(4.0, 2.0, n_fraud)).clip(0.5, 10000)
    amount_all = np.concatenate([amount_normal, amount_fraud])

    # Class labels
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    # Assemble DataFrame and shuffle
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time_all, features_all, amount_all, labels])
    df = pd.DataFrame(data, columns=columns)

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    return df


def display_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the generated dataset."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples : {len(df):,}")
    print(f"Features      : {df.shape[1] - 1}")
    print(f"Normal (0)    : {(df['Class'] == 0).sum():,}  ({(df['Class'] == 0).mean() * 100:.1f}%)")
    print(f"Fraud  (1)    : {(df['Class'] == 1).sum():,}  ({(df['Class'] == 1).mean() * 100:.1f}%)")

    print("\nAmount statistics by class:")
    print(df.groupby("Class")["Amount"].describe().round(2))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes.value_counts())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "creditcard.csv")

    df = generate_synthetic_data(
        n_samples=100000,
        fraud_ratio=0.02,
        random_state=42,
        output_path=output_path,
    )

    display_statistics(df)

    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **Imbalanced classes** — Only 2 % of transactions are fraudulent, mirroring
  real-world credit card fraud rates.
- **Shifted means for fraud** — `fraud_shift = rng.uniform(-3, 3, 28)` ensures
  the model can learn to distinguish classes.
- **Reproducibility** — A fixed `random_state` makes the dataset deterministic.

---

### `src/batch/train_model.py` — Model Training Pipeline

Provides the `FraudDetectionTrainer` class that orchestrates the full training
lifecycle: loading data, splitting / scaling features, training a Logistic
Regression classifier with balanced class weights, evaluating with ROC-AUC /
F1 / confusion matrix / cross-validation, and saving the model and scaler as
versioned `joblib` artefacts.

```python
"""
Fraud Detection Model Training Pipeline.

This script provides a complete, production-ready training pipeline for a
credit card fraud detection model using Logistic Regression.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """
    End-to-end trainer for the credit card fraud detection model.

    Attributes
    ----------
    data_path : str
        Path to the CSV dataset.
    model_dir : str
        Directory to save trained models.
    test_size : float
        Fraction of data reserved for testing (0 < test_size < 1).
    random_state : int
        Random seed used throughout the pipeline.
    """

    FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    TARGET_COLUMN = "Class"

    def __init__(
        self,
        data_path: str,
        model_dir: str = "models",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.model_dir = model_dir
        self.test_size = test_size
        self.random_state = random_state

        self.df: pd.DataFrame = None
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        self.scaler: StandardScaler = None
        self.model: LogisticRegression = None

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from *self.data_path*."""
        logger.info("Loading data from %s", self.data_path)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        logger.info("Loaded %d rows × %d columns", *self.df.shape)

        missing = [c for c in self.FEATURE_COLUMNS + [self.TARGET_COLUMN] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        fraud_count = self.df[self.TARGET_COLUMN].sum()
        logger.info(
            "Class distribution — Normal: %d (%.1f%%), Fraud: %d (%.1f%%)",
            len(self.df) - fraud_count,
            (1 - fraud_count / len(self.df)) * 100,
            fraud_count,
            fraud_count / len(self.df) * 100,
        )
        return self.df

    def prepare_data(self):
        """Split and scale features using stratified 80/20 split."""
        if self.df is None:
            raise RuntimeError("Call load_data() before prepare_data().")

        logger.info("Preparing data (test_size=%.0f%%)", self.test_size * 100)
        X = self.df[self.FEATURE_COLUMNS].values
        y = self.df[self.TARGET_COLUMN].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info("Train: %d samples | Test: %d samples", len(self.X_train), len(self.X_test))

    def train_model(self) -> LogisticRegression:
        """Train a Logistic Regression model with balanced class weights."""
        if self.X_train is None:
            raise RuntimeError("Call prepare_data() before train_model().")

        logger.info("Training Logistic Regression (class_weight='balanced') …")
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training complete.")
        return self.model

    def evaluate_model(self) -> dict:
        """Evaluate the trained model with ROC-AUC, F1, confusion matrix, cross-validation."""
        if self.model is None:
            raise RuntimeError("Call train_model() before evaluate_model().")

        logger.info("Evaluating model on test set …")
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        roc_auc = roc_auc_score(self.y_test, y_prob)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv, scoring="roc_auc"
        )

        metrics = {
            "roc_auc": roc_auc,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
        }

        self._print_evaluation(metrics, cm, report)
        return metrics

    def save_model(self, metrics: dict = None) -> dict:
        """Save model, scaler, and optional metrics as versioned joblib artefacts."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Train the model before saving.")

        os.makedirs(self.model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        paths = {
            "model": os.path.join(self.model_dir, "fraud_model.pkl"),
            "scaler": os.path.join(self.model_dir, "scaler.pkl"),
            "model_versioned": os.path.join(self.model_dir, f"fraud_model_{timestamp}.pkl"),
            "scaler_versioned": os.path.join(self.model_dir, f"scaler_{timestamp}.pkl"),
        }

        joblib.dump(self.model, paths["model"])
        joblib.dump(self.scaler, paths["scaler"])
        joblib.dump(self.model, paths["model_versioned"])
        joblib.dump(self.scaler, paths["scaler_versioned"])

        logger.info("Model saved → %s", paths["model"])
        logger.info("Scaler saved → %s", paths["scaler"])
        logger.info("Versioned backup → %s", paths["model_versioned"])

        if metrics:
            meta_path = os.path.join(self.model_dir, f"metrics_{timestamp}.json")
            import json
            with open(meta_path, "w") as fh:
                json.dump(
                    {
                        "timestamp": timestamp,
                        "roc_auc": metrics.get("roc_auc"),
                        "f1_score": metrics.get("f1_score"),
                        "cv_roc_auc_mean": metrics.get("cv_roc_auc_mean"),
                    },
                    fh,
                    indent=2,
                )
            paths["metrics"] = meta_path

        return paths

    def run_training_pipeline(self) -> dict:
        """Execute the complete training pipeline: load → prepare → train → evaluate → save."""
        logger.info("=" * 60)
        logger.info("FRAUD DETECTION TRAINING PIPELINE")
        logger.info("=" * 60)

        self.load_data()
        self.prepare_data()
        self.train_model()
        metrics = self.evaluate_model()
        paths = self.save_model(metrics)

        logger.info("=" * 60)
        logger.info("Pipeline complete. ROC-AUC: %.4f | F1: %.4f", metrics["roc_auc"], metrics["f1_score"])
        logger.info("=" * 60)

        return {"metrics": metrics, "paths": paths}

    def _print_evaluation(self, metrics: dict, cm: np.ndarray, report: dict) -> None:
        logger.info("\nClassification Report:\n%s", classification_report(self.y_test, self.model.predict(self.X_test)))
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("ROC-AUC Score: %.4f", metrics["roc_auc"])
        logger.info("F1 Score: %.4f", metrics["f1_score"])
        logger.info(
            "Cross-Validation ROC-AUC: %.4f ± %.4f",
            metrics["cv_roc_auc_mean"],
            metrics["cv_roc_auc_std"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--data-path",
        default=os.path.join(os.path.dirname(__file__), "../../data/creditcard.csv"),
        help="Path to the creditcard.csv dataset.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.dirname(__file__), "../../models"),
        help="Directory where trained models will be saved.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data used for testing (default: 0.2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = FraudDetectionTrainer(
        data_path=args.data_path,
        model_dir=args.model_dir,
        test_size=args.test_size,
    )
    result = trainer.run_training_pipeline()
    return result


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **`class_weight='balanced'`** — Automatically up-weights the minority (fraud)
  class so the model does not simply predict "normal" for every transaction.
- **Stratified split** — Preserves the original fraud ratio in both train and
  test sets.
- **Versioned artefacts** — Every training run produces timestamped backups so
  you can roll back to a previous model.
- **Cross-validation** — 5-fold stratified CV on the training set gives a
  reliable estimate of generalisation performance.

---

### `src/batch/predict.py` — Inference Helper

Provides the `FraudPredictor` class that loads a saved model and scaler to run
single-transaction or batch predictions. Each prediction returns the binary
class (`0` or `1`) and the fraud probability. Includes a CLI that accepts an
input CSV, produces an output CSV, or demonstrates a single-transaction example.

```python
"""
Fraud Detection Predictor.

Load a saved model and scaler to make single or batch predictions.
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


class FraudPredictor:
    """
    Load a trained fraud detection model and scaler for inference.

    Parameters
    ----------
    model_path : str
        Path to the serialised LogisticRegression model (joblib).
    scaler_path : str
        Path to the serialised StandardScaler (joblib).
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load()

    def _load(self) -> None:
        """Load model and scaler from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info("Model loaded from %s", self.model_path)
        logger.info("Scaler loaded from %s", self.scaler_path)

    def predict_single(self, features: dict) -> dict:
        """
        Predict fraud for a single transaction.

        Parameters
        ----------
        features : dict
            Dictionary mapping feature names (V1–V28, Amount, Time) to values.

        Returns
        -------
        dict
            {"prediction": int, "fraud_probability": float}
        """
        row = pd.DataFrame([features])[FEATURE_COLUMNS]
        scaled = self.scaler.transform(row)
        prediction = int(self.model.predict(scaled)[0])
        probability = float(self.model.predict_proba(scaled)[0][1])
        return {"prediction": prediction, "fraud_probability": round(probability, 6)}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for a DataFrame of transactions.

        Returns the input DataFrame with two new columns:
        ``prediction`` and ``fraud_probability``.
        """
        result = df.copy()
        X = result[FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X)
        result["prediction"] = self.model.predict(X_scaled).astype(int)
        result["fraud_probability"] = self.model.predict_proba(X_scaled)[:, 1].round(6)
        return result

    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """Load a CSV file, run batch predictions, and optionally save results."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s", len(df), csv_path)

        predictions = self.predict_batch(df)
        fraud_count = predictions["prediction"].sum()
        logger.info(
            "Predicted %d fraud transactions out of %d (%.2f%%)",
            fraud_count, len(predictions), fraud_count / len(predictions) * 100,
        )

        if output_path:
            predictions.to_csv(output_path, index=False)
            logger.info("Predictions saved to %s", output_path)

        return predictions


def parse_args():
    default_model = os.path.join(os.path.dirname(__file__), "../../models/fraud_model.pkl")
    default_scaler = os.path.join(os.path.dirname(__file__), "../../models/scaler.pkl")

    parser = argparse.ArgumentParser(description="Run fraud detection predictions.")
    parser.add_argument("--model", default=default_model, help="Path to the model .pkl file.")
    parser.add_argument("--scaler", default=default_scaler, help="Path to the scaler .pkl file.")
    parser.add_argument("--input", help="Path to input CSV for batch predictions.")
    parser.add_argument("--output", help="Path to save prediction results CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        predictor = FraudPredictor(model_path=args.model, scaler_path=args.scaler)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error("Run train_model.py first to generate model artefacts.")
        sys.exit(1)

    if args.input:
        predictions = predictor.predict_from_csv(args.input, output_path=args.output)
        print(predictions[["prediction", "fraud_probability"]].head(10))
        return

    # Example usage with a single synthetic transaction
    logger.info("No --input provided; running example single-transaction prediction.")
    rng = np.random.RandomState(0)
    example_features = {f"V{i}": float(rng.randn()) for i in range(1, 29)}
    example_features["Amount"] = 125.50
    example_features["Time"] = 43200.0

    result = predictor.predict_single(example_features)
    label = "FRAUD" if result["prediction"] == 1 else "NORMAL"
    print(f"\nExample transaction → {label} (probability: {result['fraud_probability']:.4f})")


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **Eager loading** — Model and scaler are loaded at construction time so that
  prediction calls are fast.
- **`predict_single`** returns a plain dict, making it easy to integrate with
  REST APIs or Kafka consumers.
- **`predict_batch`** preserves the original DataFrame columns so downstream
  consumers can join predictions back to raw data.

---

### `src/batch/batch_processor.py` — Batch Processing Pipeline

Provides the `BatchProcessor` class for processing historical transaction data.
Loads data from CSV or Parquet, applies the fraud model in configurable chunks
(default 10 000 rows) for memory efficiency, writes results to CSV or Parquet
with optional partitioning, and generates a summary report.

```python
"""
Batch processor for historical credit card fraud detection.

Loads transaction data from CSV or Parquet, applies the trained fraud
detection model, and writes results with proper partitioning.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


class BatchProcessor:
    """
    Process historical transaction data with a trained fraud detection model.

    Supports CSV and Parquet input sources and can write results to CSV
    or Parquet output with optional partitioning.

    Parameters
    ----------
    model_path : str
        Path to the serialised model (joblib).
    scaler_path : str
        Path to the serialised scaler (joblib).
    output_dir : str
        Directory where prediction results are written.
    chunk_size : int
        Number of rows processed per chunk (for memory efficiency).
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        output_dir: str = "batch_results",
        chunk_size: int = 10000,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.model = None
        self.scaler = None

    def load_model(self) -> None:
        """Load model and scaler artefacts from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info("Model loaded from %s", self.model_path)
        logger.info("Scaler loaded from %s", self.scaler_path)

    def load_data(self, path: str, fmt: str = "csv") -> pd.DataFrame:
        """Load transaction data from CSV or Parquet."""
        logger.info("Loading data from %s (format=%s)", path, fmt)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path not found: {path}")

        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'csv' or 'parquet'.")

        logger.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fraud detection model to *df* in chunks for memory efficiency.

        Returns the input DataFrame with ``prediction`` and ``fraud_probability`` columns.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call load_model() before process().")

        results = []
        n_chunks = max(1, len(df) // self.chunk_size)
        logger.info("Processing %d rows in %d chunk(s) …", len(df), n_chunks)

        for i, chunk in enumerate(self._iterate_chunks(df)):
            X = chunk[FEATURE_COLUMNS].values
            X_scaled = self.scaler.transform(X)
            chunk = chunk.copy()
            chunk["prediction"] = self.model.predict(X_scaled).astype(int)
            chunk["fraud_probability"] = self.model.predict_proba(X_scaled)[:, 1].round(6)
            results.append(chunk)

            if (i + 1) % 10 == 0:
                logger.info("  … processed %d/%d chunks", i + 1, n_chunks)

        return pd.concat(results, ignore_index=True)

    def _iterate_chunks(self, df: pd.DataFrame):
        """Yield DataFrame chunks of *self.chunk_size* rows."""
        for start in range(0, len(df), self.chunk_size):
            yield df.iloc[start : start + self.chunk_size]

    def save_results(self, results: pd.DataFrame, fmt: str = "csv", partition_by: str = None) -> str:
        """Save prediction results to *self.output_dir* in CSV or Parquet."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if fmt == "parquet":
            if partition_by and partition_by in results.columns:
                out_path = os.path.join(self.output_dir, f"predictions_{timestamp}")
                os.makedirs(out_path, exist_ok=True)
                for value, group in results.groupby(partition_by):
                    part_dir = os.path.join(out_path, f"{partition_by}={value}")
                    os.makedirs(part_dir, exist_ok=True)
                    group.to_parquet(os.path.join(part_dir, "part-0.parquet"), index=False)
            else:
                out_path = os.path.join(self.output_dir, f"predictions_{timestamp}.parquet")
                results.to_parquet(out_path, index=False)
        else:
            out_path = os.path.join(self.output_dir, f"predictions_{timestamp}.csv")
            results.to_csv(out_path, index=False)

        logger.info("Results saved to %s", out_path)
        return out_path

    def generate_report(self, results: pd.DataFrame) -> dict:
        """Generate a summary report with fraud counts, risk breakdowns, and averages."""
        total = len(results)
        fraud_count = int(results["prediction"].sum())
        normal_count = total - fraud_count

        high_risk = results[results["fraud_probability"] >= 0.8]
        medium_risk = results[
            (results["fraud_probability"] >= 0.5) & (results["fraud_probability"] < 0.8)
        ]

        report = {
            "total_transactions": total,
            "fraud_detected": fraud_count,
            "normal_transactions": normal_count,
            "fraud_rate_pct": round(fraud_count / total * 100, 2),
            "high_risk_count": len(high_risk),
            "medium_risk_count": len(medium_risk),
            "avg_fraud_probability": round(float(results["fraud_probability"].mean()), 4),
        }

        logger.info(
            "\nBatch Report:\n"
            "  Total transactions : %d\n"
            "  Fraud detected     : %d (%.2f%%)\n"
            "  High risk (≥0.8)  : %d\n"
            "  Medium risk (≥0.5): %d",
            total, fraud_count, report["fraud_rate_pct"], len(high_risk), len(medium_risk),
        )
        return report

    def run(self, input_path: str, input_fmt: str = "csv", output_fmt: str = "csv",
            partition_by: str = None) -> dict:
        """Run the full batch pipeline: load model → load data → process → save → report."""
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING PIPELINE")
        logger.info("=" * 60)

        self.load_model()
        df = self.load_data(input_path, fmt=input_fmt)
        results = self.process(df)
        out_path = self.save_results(results, fmt=output_fmt, partition_by=partition_by)
        report = self.generate_report(results)
        report["output_path"] = out_path

        logger.info("Pipeline complete.")
        return report


def parse_args():
    default_model = os.path.join(os.path.dirname(__file__), "../../models/fraud_model.pkl")
    default_scaler = os.path.join(os.path.dirname(__file__), "../../models/scaler.pkl")

    parser = argparse.ArgumentParser(description="Batch process transactions for fraud detection.")
    parser.add_argument("--model", default=default_model, help="Path to model .pkl file.")
    parser.add_argument("--scaler", default=default_scaler, help="Path to scaler .pkl file.")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "../../data/creditcard.csv"),
        help="Path to input data file.",
    )
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Input format.")
    parser.add_argument("--output-dir", default="batch_results", help="Output directory.")
    parser.add_argument(
        "--output-format", default="csv", choices=["csv", "parquet"], help="Output format."
    )
    parser.add_argument("--partition-by", default=None, help="Column for Parquet partitioning.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Rows per processing chunk.")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = BatchProcessor(
        model_path=args.model,
        scaler_path=args.scaler,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )
    try:
        report = processor.run(
            input_path=args.input,
            input_fmt=args.format,
            output_fmt=args.output_format,
            partition_by=args.partition_by,
        )
        print("\nBatch report:", report)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **Chunked processing** — Processes data in configurable chunks (`chunk_size`)
  so that large datasets do not exhaust memory.
- **Format flexibility** — Reads from and writes to both CSV and Parquet.
- **Partitioned Parquet output** — When `partition_by` is set, the output is
  organised into Hive-style `column=value/` directories.
- **Report generation** — The summary dict includes total/fraud/normal counts,
  fraud rate, risk tier breakdowns, and average fraud probability.

---

### `src/streaming/fraud_detector.py` — Basic Spark Streaming Job

Reads JSON-encoded transaction messages from the Kafka `transactions` topic,
applies the trained model via a Pandas UDF (vectorised for efficiency), and
writes results to three sinks: console (live monitoring), Parquet (persistent
storage), and the Kafka `fraud_alerts` topic (downstream alerting). The model
and scaler are broadcast to all Spark executors.

```python
"""
Real-time Spark Structured Streaming fraud detector.

Reads transaction JSON messages from the Kafka topic ``transactions``,
applies a trained fraud detection model, and writes results to:

  * Console (monitoring)
  * Parquet files (persistent storage)
  * Kafka topic ``fraud_alerts`` (downstream alerting)
"""

import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "fraud_alerts")
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/tmp/fraud_detector_checkpoint")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/fraud_detector_output")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load the trained model and scaler from disk."""
    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Loaded model from %s", model_path)
    logger.info("Loaded scaler from %s", scaler_path)
    return model, scaler


def build_schema():
    """Return the StructType schema for incoming transaction JSON messages."""
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    fields = [StructField("TransactionID", StringType(), True)]
    for col in FEATURE_COLUMNS:
        fields.append(StructField(col, DoubleType(), True))
    fields.append(StructField("Timestamp", DoubleType(), True))
    return StructType(fields)


def build_spark_session(app_name: str = "FraudDetector"):
    """Create and return a SparkSession with Kafka-compatible settings."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_kafka_stream(spark):
    """Return a streaming DataFrame reading from the transactions Kafka topic."""
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_transactions(raw_df, schema):
    """Parse the raw Kafka binary value column into structured columns."""
    from pyspark.sql.functions import col, from_json

    return (
        raw_df.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
        )
        .select("data.*", "kafka_timestamp")
    )


def apply_fraud_model(transactions_df, model, scaler):
    """
    Apply the fraud detection model to the streaming DataFrame.

    Uses a Pandas UDF (vectorised) so the model runs efficiently across
    Spark partitions.
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType
    import pandas as pd

    # Broadcast the model and scaler objects to all executors
    sc = transactions_df.sparkSession.sparkContext
    bc_model = sc.broadcast(model)
    bc_scaler = sc.broadcast(scaler)

    @pandas_udf(DoubleType())
    def predict_fraud_probability(*cols):
        X = pd.concat(list(cols), axis=1)
        X.columns = FEATURE_COLUMNS
        X_scaled = bc_scaler.value.transform(X.values)
        probs = bc_model.value.predict_proba(X_scaled)[:, 1]
        return pd.Series(probs)

    feature_cols = [transactions_df[c] for c in FEATURE_COLUMNS]
    return transactions_df.withColumn("fraud_probability", predict_fraud_probability(*feature_cols))


def add_prediction_flag(df):
    """Add a binary ``is_fraud`` column (1 when fraud_probability >= 0.5)."""
    from pyspark.sql.functions import col, when

    return df.withColumn(
        "is_fraud",
        when(col("fraud_probability") >= 0.5, 1).otherwise(0),
    )


def write_to_console(df, checkpoint_dir: str):
    """Write stream to console for monitoring (truncated output)."""
    return (
        df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "console"))
        .start()
    )


def write_to_parquet(df, output_dir: str, checkpoint_dir: str):
    """Write stream to Parquet files for persistent storage."""
    return (
        df.writeStream.outputMode("append")
        .format("parquet")
        .option("path", output_dir)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet"))
        .trigger(processingTime="30 seconds")
        .start()
    )


def write_fraud_alerts_to_kafka(df, checkpoint_dir: str):
    """Forward fraud alerts to the fraud_alerts Kafka topic as JSON."""
    from pyspark.sql.functions import col, struct, to_json

    alert_df = df.filter(col("is_fraud") == 1).select(
        to_json(
            struct(col("TransactionID"), col("fraud_probability"), col("kafka_timestamp"))
        ).alias("value")
    )

    return (
        alert_df.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", KAFKA_OUTPUT_TOPIC)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka"))
        .start()
    )


def main():
    logger.info("Starting Fraud Detector streaming job …")
    logger.info("  Kafka bootstrap : %s", KAFKA_BOOTSTRAP_SERVERS)
    logger.info("  Input topic     : %s", KAFKA_INPUT_TOPIC)
    logger.info("  Output topic    : %s", KAFKA_OUTPUT_TOPIC)

    try:
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    except FileNotFoundError as exc:
        logger.error("%s — train the model first with src/batch/train_model.py", exc)
        sys.exit(1)

    spark = build_spark_session()
    schema = build_schema()

    raw_stream = read_kafka_stream(spark)
    transactions = parse_transactions(raw_stream, schema)
    scored = apply_fraud_model(transactions, model, scaler)
    final = add_prediction_flag(scored)

    queries = [
        write_to_console(final, CHECKPOINT_DIR),
        write_to_parquet(final, OUTPUT_DIR, CHECKPOINT_DIR),
        write_fraud_alerts_to_kafka(final, CHECKPOINT_DIR),
    ]

    logger.info("Streaming queries started. Awaiting termination …")
    for q in queries:
        q.awaitTermination()


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **Pandas UDF (vectorised)** — `@pandas_udf` applies the scikit-learn model
  in vectorised batches, avoiding the overhead of row-by-row Python calls.
- **Broadcast variables** — The model and scaler are broadcast via Spark's
  `sc.broadcast()` so each executor receives a single copy rather than
  serialising per task.
- **Three output sinks** — Console for live debugging, Parquet for durable
  storage, and Kafka for real-time downstream alerting.

---

### `src/streaming/fraud_detector_enhanced.py` — Enhanced Streaming Detector

Extends the basic detector with real-time feature engineering: transaction
velocity per user, per-user amount aggregations (mean, max, std), and hour-of-day
extraction. It uses Spark's windowed aggregations with watermarking for
fault-tolerant stateful processing and assigns a three-tier risk label (HIGH /
MEDIUM / LOW) based on the predicted fraud probability.

```python
"""
Enhanced Real-time Spark Structured Streaming fraud detector.

Extends the basic ``fraud_detector.py`` with real-time feature engineering:

  * Transaction velocity  (transactions per user in a sliding window)
  * Amount aggregations   (mean, max, std per user window)
  * Time-based features   (hour of day from Unix timestamp)

Supports two output modes:
  * ``append``  — new fraud detections
  * ``update``  — aggregated per-user statistics
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "fraud_alerts")
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/tmp/fraud_enhanced_checkpoint")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/fraud_enhanced_output")
WINDOW_DURATION = os.getenv("WINDOW_DURATION", "10 minutes")
SLIDE_DURATION = os.getenv("SLIDE_DURATION", "5 minutes")
WATERMARK_DELAY = os.getenv("WATERMARK_DELAY", "5 minutes")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load and return the trained model and scaler."""
    import joblib

    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefact not found: {path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model loaded: %s", model_path)
    logger.info("Scaler loaded: %s", scaler_path)
    return model, scaler


def build_schema():
    """Build the StructType for incoming transaction JSON."""
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    fields = [
        StructField("TransactionID", StringType(), True),
        StructField("UserID", StringType(), True),
    ]
    for col_name in FEATURE_COLUMNS:
        fields.append(StructField(col_name, DoubleType(), True))
    fields.append(StructField("Timestamp", DoubleType(), True))
    return StructType(fields)


def build_spark_session(app_name: str = "FraudDetectorEnhanced"):
    """Create a SparkSession with optimised settings."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.stateStore.providerClass",
                "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Feature engineering

def add_time_features(df):
    """Extract hour-of-day from the Unix ``Timestamp`` column."""
    from pyspark.sql.functions import col, from_unixtime, hour

    return df.withColumn(
        "event_time", from_unixtime(col("Timestamp").cast("long"))
    ).withColumn("hour_of_day", hour("event_time"))


def add_windowed_features(df):
    """
    Compute per-user windowed aggregations:

    * ``tx_count``    — transaction count in the window
    * ``amount_mean`` — mean Amount per user
    * ``amount_max``  — max Amount per user
    * ``amount_std``  — std dev of Amount per user
    """
    from pyspark.sql.functions import avg, col, count, max as spark_max, stddev, window

    agg_df = (
        df.withWatermark("event_time", WATERMARK_DELAY)
        .groupBy(
            col("UserID"),
            window(col("event_time"), WINDOW_DURATION, SLIDE_DURATION),
        )
        .agg(
            count("TransactionID").alias("tx_count"),
            avg("Amount").alias("amount_mean"),
            spark_max("Amount").alias("amount_max"),
            stddev("Amount").alias("amount_std"),
        )
        .withColumn("window_start", col("window.start"))
        .withColumn("window_end", col("window.end"))
        .drop("window")
    )
    return agg_df


# Model application

def apply_fraud_model(df, model, scaler):
    """Apply the fraud model via a Pandas UDF (vectorised)."""
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    sc = df.sparkSession.sparkContext
    bc_model = sc.broadcast(model)
    bc_scaler = sc.broadcast(scaler)

    @pandas_udf(DoubleType())
    def predict_probability(*cols):
        X = pd.concat(list(cols), axis=1)
        X.columns = FEATURE_COLUMNS
        X_scaled = bc_scaler.value.transform(X.fillna(0).values)
        probs = bc_model.value.predict_proba(X_scaled)[:, 1]
        return pd.Series(probs)

    feature_cols = [df[c] for c in FEATURE_COLUMNS]
    return df.withColumn("fraud_probability", predict_probability(*feature_cols))


def add_risk_label(df):
    """Assign a string risk label based on fraud_probability."""
    from pyspark.sql.functions import col, when

    return df.withColumn(
        "risk_label",
        when(col("fraud_probability") >= 0.8, "HIGH")
        .when(col("fraud_probability") >= 0.5, "MEDIUM")
        .otherwise("LOW"),
    ).withColumn(
        "is_fraud",
        when(col("fraud_probability") >= 0.5, 1).otherwise(0),
    )


# Output sinks

def write_fraud_detections(df, checkpoint_dir: str, output_dir: str):
    """Write new fraud detections (append mode) to Parquet and console."""
    from pyspark.sql.functions import col

    fraud_df = df.filter(col("is_fraud") == 1)

    parquet_query = (
        fraud_df.writeStream.outputMode("append")
        .format("parquet")
        .option("path", os.path.join(output_dir, "fraud_detections"))
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet_detections"))
        .trigger(processingTime="30 seconds")
        .start()
    )

    console_query = (
        fraud_df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoint_dir, "console"))
        .start()
    )

    return [parquet_query, console_query]


def write_aggregated_stats(agg_df, checkpoint_dir: str, output_dir: str):
    """Write per-user windowed aggregations (update mode) to Parquet."""
    return (
        agg_df.writeStream.outputMode("update")
        .format("parquet")
        .option("path", os.path.join(output_dir, "user_stats"))
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet_stats"))
        .trigger(processingTime="60 seconds")
        .start()
    )


def write_alerts_to_kafka(df, checkpoint_dir: str):
    """Write high-risk alerts back to Kafka as JSON."""
    from pyspark.sql.functions import col, struct, to_json

    alert_df = (
        df.filter(col("risk_label") == "HIGH")
        .select(
            to_json(
                struct(
                    col("TransactionID"),
                    col("UserID"),
                    col("fraud_probability"),
                    col("risk_label"),
                    col("event_time"),
                )
            ).alias("value")
        )
    )

    return (
        alert_df.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", KAFKA_OUTPUT_TOPIC)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka"))
        .start()
    )


def main():
    logger.info("Starting Enhanced Fraud Detector …")
    logger.info("  Kafka bootstrap : %s", KAFKA_BOOTSTRAP_SERVERS)
    logger.info("  Input topic     : %s", KAFKA_INPUT_TOPIC)
    logger.info("  Output topic    : %s", KAFKA_OUTPUT_TOPIC)
    logger.info("  Window          : %s / slide %s", WINDOW_DURATION, SLIDE_DURATION)

    try:
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    except FileNotFoundError as exc:
        logger.error("%s — run src/batch/train_model.py first", exc)
        sys.exit(1)

    from pyspark.sql.functions import col, from_json

    spark = build_spark_session()
    schema = build_schema()

    # Read from Kafka
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON
    transactions = (
        raw_stream.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_ts"),
        )
        .select("data.*", "kafka_ts")
    )

    # Feature engineering
    transactions = add_time_features(transactions)

    # Score transactions
    scored = apply_fraud_model(transactions, model, scaler)
    final = add_risk_label(scored)

    # Windowed aggregations
    agg_stats = add_windowed_features(transactions)

    # Write outputs
    queries = []
    queries.extend(write_fraud_detections(final, CHECKPOINT_DIR, OUTPUT_DIR))
    queries.append(write_aggregated_stats(agg_stats, CHECKPOINT_DIR, OUTPUT_DIR))
    queries.append(write_alerts_to_kafka(final, CHECKPOINT_DIR))

    logger.info("%d streaming queries started. Awaiting termination …", len(queries))
    for q in queries:
        q.awaitTermination()


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **Watermarking** — `withWatermark("event_time", "5 minutes")` tells Spark to
  discard state for events older than 5 minutes, keeping memory bounded.
- **Sliding window aggregations** — A 10-minute window sliding every 5 minutes
  computes per-user transaction velocity and amount statistics.
- **Three-tier risk labels** — HIGH (≥ 0.8), MEDIUM (≥ 0.5), LOW (< 0.5) help
  downstream consumers prioritise alerts.
- **NaN handling** — `X.fillna(0)` in the Pandas UDF prevents crashes on
  partial data.

---

### `scripts/kafka_producer.py` — Kafka Transaction Producer

Simulates a real-time stream of credit card transactions by generating random
JSON-encoded messages and publishing them to a Kafka topic at a configurable
rate. Each message contains a transaction ID, user ID, timestamp, amount,
location, merchant type, and all 28 anonymous features.

```python
"""
Kafka producer that simulates real-time credit card transactions.

Sends JSON-encoded transaction messages to the ``transactions`` Kafka topic
at a configurable rate.
"""

import argparse
import json
import logging
import random
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
MERCHANT_TYPES = ["online", "retail", "restaurant", "grocery", "travel", "entertainment"]


def generate_transaction(rng: random.Random) -> dict:
    """Generate a single synthetic transaction record."""
    features = {f"V{i}": round(rng.gauss(0, 1), 6) for i in range(1, 29)}
    return {
        "TransactionID": f"TXN{rng.randint(100000, 999999)}",
        "UserID": f"USER{rng.randint(1, 1000):04d}",
        "Timestamp": time.time(),
        "Amount": round(rng.lognormvariate(3.5, 1.5), 2),
        "Location": rng.choice(LOCATIONS),
        "MerchantType": rng.choice(MERCHANT_TYPES),
        **features,
        "Time": time.time() % 172800,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Send synthetic transactions to Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                        help="Kafka bootstrap servers.")
    parser.add_argument("--topic", default="transactions",
                        help="Kafka topic to produce to.")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Transactions per second (default: 1).")
    parser.add_argument("--count", type=int, default=0,
                        help="Stop after this many messages (0 = run forever).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from kafka import KafkaProducer
    except ImportError:
        logger.error("kafka-python is not installed. Run: pip install kafka-python")
        sys.exit(1)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    rng = random.Random(args.seed)
    delay = 1.0 / max(args.rate, 0.001)
    sent = 0

    logger.info("Producing to topic '%s' on %s at %.1f msg/s …",
                args.topic, args.bootstrap_servers, args.rate)

    try:
        while args.count == 0 or sent < args.count:
            tx = generate_transaction(rng)
            producer.send(args.topic, tx)
            sent += 1
            if sent % 100 == 0:
                logger.info("Sent %d transactions", sent)
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Producer stopped by user after %d messages.", sent)
    finally:
        producer.flush()
        producer.close()
        logger.info("Total messages sent: %d", sent)


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- **Configurable rate** — `--rate` controls transactions per second; useful for
  load testing.
- **Bounded mode** — `--count N` stops after *N* messages; `--count 0` runs
  indefinitely until Ctrl-C.
- **Reproducible randomness** — Optional `--seed` produces the same sequence of
  transactions every time.

---

### `scripts/start_streaming.sh` — Start Streaming Pipeline

Orchestrates the startup of all streaming components: verifies Kafka
connectivity, starts the Kafka producer in the background, and submits the
Spark Structured Streaming job via `spark-submit`. Supports flags for the
enhanced detector (`--enhanced`) and skipping the producer (`--no-producer`).

```bash
#!/usr/bin/env bash
# Start all components of the Fraud Detection Streaming Pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables
source "${PROJECT_ROOT}/env/config.env"

# ---- Defaults ---------------------------------------------------------------
ENHANCED=false
START_PRODUCER=true
LOG_FILE="${PROJECT_ROOT}/logs/streaming.log"
PID_DIR="${PROJECT_ROOT}/.pids"

# ---- Argument parsing -------------------------------------------------------
for arg in "$@"; do
  case "${arg}" in
    --enhanced)    ENHANCED=true ;;
    --no-producer) START_PRODUCER=false ;;
    --help|-h)
      echo "Usage: $0 [--enhanced] [--no-producer]"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

# ---- Helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
fail() { log "ERROR: $*"; exit 1; }

mkdir -p "${PROJECT_ROOT}/logs" "${PID_DIR}"

# ---- 1. Check Kafka ---------------------------------------------------------
log "Checking Kafka connectivity at ${KAFKA_BOOTSTRAP_SERVERS} …"

KAFKA_HOST="${KAFKA_BOOTSTRAP_SERVERS%%:*}"
KAFKA_PORT="${KAFKA_BOOTSTRAP_SERVERS##*:}"

if ! nc -z -w 5 "${KAFKA_HOST}" "${KAFKA_PORT}" 2>/dev/null; then
  fail "Cannot reach Kafka at ${KAFKA_BOOTSTRAP_SERVERS}. Is the broker running?"
fi
log "Kafka is reachable ✓"

# ---- 2. Start Kafka producer (optional) ------------------------------------
if [ "${START_PRODUCER}" = true ]; then
  PRODUCER_SCRIPT="${PROJECT_ROOT}/scripts/kafka_producer.py"

  if [ ! -f "${PRODUCER_SCRIPT}" ]; then
    log "WARNING: Kafka producer not found at ${PRODUCER_SCRIPT}; skipping."
  else
    log "Starting Kafka producer …"
    nohup python3 "${PRODUCER_SCRIPT}" \
      >> "${PROJECT_ROOT}/logs/producer.log" 2>&1 &
    PRODUCER_PID=$!
    echo "${PRODUCER_PID}" > "${PID_DIR}/producer.pid"
    log "Kafka producer started (PID=${PRODUCER_PID})"
    sleep 2  # give the producer a moment to connect
  fi
fi

# ---- 3. Submit Spark Streaming job -----------------------------------------
if [ "${ENHANCED}" = true ]; then
  DETECTOR_SCRIPT="${PROJECT_ROOT}/src/streaming/fraud_detector_enhanced.py"
  APP_NAME="FraudDetectorEnhanced"
else
  DETECTOR_SCRIPT="${PROJECT_ROOT}/src/streaming/fraud_detector.py"
  APP_NAME="FraudDetector"
fi

if [ ! -f "${DETECTOR_SCRIPT}" ]; then
  fail "Streaming script not found: ${DETECTOR_SCRIPT}"
fi

SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"
if [ ! -x "${SPARK_SUBMIT}" ]; then
  # Try spark-submit on PATH
  SPARK_SUBMIT="spark-submit"
fi

KAFKA_PACKAGE="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

log "Submitting Spark job: ${APP_NAME}"
log "  Script  : ${DETECTOR_SCRIPT}"
log "  Package : ${KAFKA_PACKAGE}"

nohup "${SPARK_SUBMIT}" \
  --name "${APP_NAME}" \
  --properties-file "${PROJECT_ROOT}/env/spark_config.conf" \
  --packages "${KAFKA_PACKAGE}" \
  "${DETECTOR_SCRIPT}" \
  >> "${PROJECT_ROOT}/logs/spark_streaming.log" 2>&1 &

SPARK_PID=$!
echo "${SPARK_PID}" > "${PID_DIR}/spark.pid"
log "Spark streaming job started (PID=${SPARK_PID})"

log "All components started. Logs:"
log "  Producer : ${PROJECT_ROOT}/logs/producer.log"
log "  Spark    : ${PROJECT_ROOT}/logs/spark_streaming.log"
log ""
log "To stop: ./scripts/stop_streaming.sh"
```

---

### `scripts/stop_streaming.sh` — Stop Streaming Pipeline

Gracefully terminates all running streaming components by reading the PID files
written by `start_streaming.sh`. Sends `SIGTERM` first, waits up to 30 seconds,
and escalates to `SIGKILL` if the process is still alive.

```bash
#!/usr/bin/env bash
# Gracefully stop all Fraud Detection Streaming components.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_DIR="${PROJECT_ROOT}/.pids"
LOG_FILE="${PROJECT_ROOT}/logs/streaming.log"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

mkdir -p "${PROJECT_ROOT}/logs"

log "Stopping Fraud Detection Streaming components …"

# ---- Stop Spark Streaming job ----------------------------------------------
SPARK_PID_FILE="${PID_DIR}/spark.pid"
if [ -f "${SPARK_PID_FILE}" ]; then
  SPARK_PID=$(cat "${SPARK_PID_FILE}")
  if kill -0 "${SPARK_PID}" 2>/dev/null; then
    log "Sending SIGTERM to Spark job (PID=${SPARK_PID}) …"
    kill -TERM "${SPARK_PID}"
    for i in $(seq 1 30); do
      if ! kill -0 "${SPARK_PID}" 2>/dev/null; then
        log "Spark job stopped ✓"
        break
      fi
      sleep 1
    done
    if kill -0 "${SPARK_PID}" 2>/dev/null; then
      log "WARNING: Spark job did not stop gracefully; sending SIGKILL …"
      kill -9 "${SPARK_PID}" || true
    fi
  else
    log "Spark job (PID=${SPARK_PID}) is not running."
  fi
  rm -f "${SPARK_PID_FILE}"
else
  log "No Spark PID file found at ${SPARK_PID_FILE}."
fi

# ---- Stop Kafka producer ---------------------------------------------------
PRODUCER_PID_FILE="${PID_DIR}/producer.pid"
if [ -f "${PRODUCER_PID_FILE}" ]; then
  PRODUCER_PID=$(cat "${PRODUCER_PID_FILE}")
  if kill -0 "${PRODUCER_PID}" 2>/dev/null; then
    log "Stopping Kafka producer (PID=${PRODUCER_PID}) …"
    kill -TERM "${PRODUCER_PID}" || true
    sleep 2
    if kill -0 "${PRODUCER_PID}" 2>/dev/null; then
      kill -9 "${PRODUCER_PID}" || true
    fi
    log "Kafka producer stopped ✓"
  else
    log "Kafka producer (PID=${PRODUCER_PID}) is not running."
  fi
  rm -f "${PRODUCER_PID_FILE}"
else
  log "No producer PID file found at ${PRODUCER_PID_FILE}."
fi

log "All components stopped."
```

---

### `env/config.env` — Environment Variables

Central configuration file sourced before running any component. Defines Kafka
broker addresses and topic names, model artefact paths, data locations,
streaming checkpoint directories, Spark runtime settings, and window parameters
for the enhanced detector.

```bash
# Environment variables for the Fraud Detection System
# Source this file before running any component:
#   source env/config.env

# Kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_INPUT_TOPIC=transactions
export KAFKA_OUTPUT_TOPIC=fraud_alerts
export KAFKA_GROUP_ID=fraud-detector-group

# Model artefacts
export MODEL_PATH=models/fraud_model.pkl
export SCALER_PATH=models/scaler.pkl

# Data paths
export DATA_PATH=data/creditcard.csv
export BATCH_OUTPUT_DIR=batch_results

# Spark Streaming checkpoints and output
export CHECKPOINT_DIR=/tmp/fraud_detector_checkpoint
export OUTPUT_DIR=/tmp/fraud_detector_output

# Enhanced detector
export ENHANCED_CHECKPOINT_DIR=/tmp/fraud_enhanced_checkpoint
export ENHANCED_OUTPUT_DIR=/tmp/fraud_enhanced_output

# Spark
export SPARK_HOME=/opt/spark
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Window configuration (enhanced detector)
export WINDOW_DURATION="10 minutes"
export SLIDE_DURATION="5 minutes"
export WATERMARK_DELAY="5 minutes"

# Logging
export LOG_LEVEL=INFO
```

---

### `env/spark_config.conf` — Spark Configuration

Properties file passed to `spark-submit` via `--properties-file`. Configures
driver and executor memory (2 GB each), parallelism (8 partitions), Kryo
serialisation, Structured Streaming trigger interval (30 s), and Spark UI
settings.

```properties
# Spark Configuration for Fraud Detection Streaming Job

# Application
spark.app.name=FraudDetectionStreaming

# Memory settings
spark.driver.memory=2g
spark.executor.memory=2g
spark.executor.cores=2
spark.driver.maxResultSize=1g

# Memory fraction available for caching (vs. execution)
spark.memory.fraction=0.6
spark.memory.storageFraction=0.5

# Executor configuration
spark.executor.instances=2
spark.default.parallelism=8
spark.sql.shuffle.partitions=8

# Serialisation
spark.serializer=org.apache.spark.serializer.KryoSerializer
spark.kryo.registrationRequired=false

# Structured Streaming
spark.streaming.stopGracefullyOnShutdown=true
spark.sql.streaming.checkpointLocation=/tmp/fraud_detector_checkpoint

# Trigger interval (seconds) for micro-batch processing
spark.sql.streaming.trigger.processingTime=30s

# Kafka integration
spark.kafka.consumer.cache.capacity=64

# UI and logging
spark.ui.enabled=true
spark.ui.port=4040
spark.eventLog.enabled=false

# Log level for Spark internals (does not affect app logger)
spark.log.level=WARN
```

---

### `tests/test_end_to_end.py` — Integration Tests

Comprehensive test suite covering the full pipeline: data generation, model
training and evaluation, single and batch predictions, batch processor pipeline,
output format validation (CSV and Parquet), and performance smoke tests. Uses
`pytest` with session-scoped fixtures so the model is trained once and shared
across all tests. No live Kafka or Spark cluster is required.

```python
"""
End-to-end integration tests for the fraud detection pipeline.

Tests cover:
  - Data generation
  - Model training, evaluation and serialisation
  - Batch prediction (single + batch)
  - Batch processor pipeline
  - Output format validation

No live Kafka or Spark cluster is required; streaming components are
unit-tested via mocks.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure src packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generate_synthetic_data import generate_synthetic_data
from src.batch.train_model import FraudDetectionTrainer
from src.batch.predict import FraudPredictor, FEATURE_COLUMNS
from src.batch.batch_processor import BatchProcessor


# Fixtures

@pytest.fixture(scope="session")
def synthetic_df():
    """Small synthetic dataset shared across the test session."""
    return generate_synthetic_data(n_samples=1000, fraud_ratio=0.05, random_state=0)


@pytest.fixture(scope="session")
def tmp_base(tmp_path_factory):
    """Temporary directory for artefacts."""
    return tmp_path_factory.mktemp("fraud_test")


@pytest.fixture(scope="session")
def csv_path(synthetic_df, tmp_base):
    """Write the synthetic dataset to a CSV and return its path."""
    path = str(tmp_base / "creditcard.csv")
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def model_dir(tmp_base):
    """Return a temp directory for saved models."""
    d = str(tmp_base / "models")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def trained_artefacts(csv_path, model_dir):
    """Run the full training pipeline once and return the result dict."""
    trainer = FraudDetectionTrainer(
        data_path=csv_path, model_dir=model_dir, test_size=0.2,
    )
    return trainer.run_training_pipeline()


@pytest.fixture(scope="session")
def predictor(trained_artefacts):
    """Return an initialised FraudPredictor backed by the session-level model."""
    return FraudPredictor(
        model_path=trained_artefacts["paths"]["model"],
        scaler_path=trained_artefacts["paths"]["scaler"],
    )


# 1. Data generation

class TestDataGeneration:
    def test_shape(self, synthetic_df):
        assert synthetic_df.shape == (1000, 31), "Expected 1000 rows and 31 columns"

    def test_columns(self, synthetic_df):
        expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        assert list(synthetic_df.columns) == expected

    def test_fraud_ratio(self, synthetic_df):
        ratio = synthetic_df["Class"].mean()
        assert 0.01 <= ratio <= 0.20, f"Unexpected fraud ratio: {ratio}"

    def test_no_nulls(self, synthetic_df):
        assert synthetic_df.isnull().sum().sum() == 0

    def test_class_values(self, synthetic_df):
        assert set(synthetic_df["Class"].unique()).issubset({0, 1})

    def test_reproducible(self):
        df1 = generate_synthetic_data(n_samples=200, random_state=99)
        df2 = generate_synthetic_data(n_samples=200, random_state=99)
        pd.testing.assert_frame_equal(df1, df2)


# 2. Model training

class TestModelTraining:
    def test_pipeline_returns_metrics(self, trained_artefacts):
        assert "metrics" in trained_artefacts
        assert "paths" in trained_artefacts

    def test_roc_auc_reasonable(self, trained_artefacts):
        roc = trained_artefacts["metrics"]["roc_auc"]
        assert roc > 0.5, f"ROC-AUC too low: {roc}"

    def test_f1_non_negative(self, trained_artefacts):
        f1 = trained_artefacts["metrics"]["f1_score"]
        assert 0.0 <= f1 <= 1.0

    def test_model_file_exists(self, trained_artefacts):
        assert os.path.exists(trained_artefacts["paths"]["model"])

    def test_scaler_file_exists(self, trained_artefacts):
        assert os.path.exists(trained_artefacts["paths"]["scaler"])

    def test_versioned_model_exists(self, trained_artefacts):
        assert os.path.exists(trained_artefacts["paths"]["model_versioned"])

    def test_metrics_json_saved(self, trained_artefacts):
        if "metrics" in trained_artefacts["paths"]:
            with open(trained_artefacts["paths"]["metrics"]) as fh:
                meta = json.load(fh)
            assert "roc_auc" in meta

    def test_confusion_matrix_shape(self, trained_artefacts):
        cm = trained_artefacts["metrics"]["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2


# 3. Single prediction

class TestSinglePrediction:
    def _example_features(self):
        rng = np.random.RandomState(1)
        feats = {f"V{i}": float(rng.randn()) for i in range(1, 29)}
        feats["Amount"] = 50.0
        feats["Time"] = 3600.0
        return feats

    def test_returns_prediction_and_probability(self, predictor):
        result = predictor.predict_single(self._example_features())
        assert "prediction" in result
        assert "fraud_probability" in result

    def test_prediction_binary(self, predictor):
        result = predictor.predict_single(self._example_features())
        assert result["prediction"] in (0, 1)

    def test_probability_in_range(self, predictor):
        result = predictor.predict_single(self._example_features())
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_deterministic(self, predictor):
        feats = self._example_features()
        r1 = predictor.predict_single(feats)
        r2 = predictor.predict_single(feats)
        assert r1 == r2


# 4. Batch prediction

class TestBatchPrediction:
    def test_columns_added(self, predictor, synthetic_df):
        result = predictor.predict_batch(synthetic_df)
        assert "prediction" in result.columns
        assert "fraud_probability" in result.columns

    def test_row_count_preserved(self, predictor, synthetic_df):
        result = predictor.predict_batch(synthetic_df)
        assert len(result) == len(synthetic_df)

    def test_prediction_values(self, predictor, synthetic_df):
        result = predictor.predict_batch(synthetic_df)
        assert set(result["prediction"].unique()).issubset({0, 1})

    def test_probabilities_in_range(self, predictor, synthetic_df):
        result = predictor.predict_batch(synthetic_df)
        assert (result["fraud_probability"] >= 0).all()
        assert (result["fraud_probability"] <= 1).all()

    def test_predict_from_csv(self, predictor, csv_path, tmp_base):
        out = str(tmp_base / "predictions.csv")
        result = predictor.predict_from_csv(csv_path, output_path=out)
        assert os.path.exists(out)
        saved = pd.read_csv(out)
        assert len(saved) == len(result)


# 5. Batch processor

class TestBatchProcessor:
    def test_full_pipeline(self, trained_artefacts, csv_path, tmp_base):
        processor = BatchProcessor(
            model_path=trained_artefacts["paths"]["model"],
            scaler_path=trained_artefacts["paths"]["scaler"],
            output_dir=str(tmp_base / "batch_out"),
            chunk_size=200,
        )
        report = processor.run(input_path=csv_path, input_fmt="csv", output_fmt="csv")
        assert "total_transactions" in report
        assert report["total_transactions"] == 1000

    def test_report_keys(self, trained_artefacts, csv_path, tmp_base):
        processor = BatchProcessor(
            model_path=trained_artefacts["paths"]["model"],
            scaler_path=trained_artefacts["paths"]["scaler"],
            output_dir=str(tmp_base / "batch_out2"),
        )
        report = processor.run(input_path=csv_path, input_fmt="csv")
        for key in ("fraud_detected", "normal_transactions", "fraud_rate_pct", "output_path"):
            assert key in report

    def test_output_file_created(self, trained_artefacts, csv_path, tmp_base):
        out_dir = str(tmp_base / "batch_out3")
        processor = BatchProcessor(
            model_path=trained_artefacts["paths"]["model"],
            scaler_path=trained_artefacts["paths"]["scaler"],
            output_dir=out_dir,
        )
        report = processor.run(input_path=csv_path, input_fmt="csv")
        assert os.path.exists(report["output_path"])

    def test_missing_model_raises(self, tmp_base):
        processor = BatchProcessor(
            model_path="/nonexistent/model.pkl",
            scaler_path="/nonexistent/scaler.pkl",
        )
        with pytest.raises(FileNotFoundError):
            processor.load_model()


# 6. Data flow / output format

class TestOutputFormat:
    def test_csv_output_readable(self, trained_artefacts, csv_path, tmp_base):
        processor = BatchProcessor(
            model_path=trained_artefacts["paths"]["model"],
            scaler_path=trained_artefacts["paths"]["scaler"],
            output_dir=str(tmp_base / "fmt_out"),
        )
        report = processor.run(input_path=csv_path)
        result = pd.read_csv(report["output_path"])
        assert "prediction" in result.columns
        assert "fraud_probability" in result.columns

    def test_parquet_output_readable(self, trained_artefacts, csv_path, tmp_base):
        processor = BatchProcessor(
            model_path=trained_artefacts["paths"]["model"],
            scaler_path=trained_artefacts["paths"]["scaler"],
            output_dir=str(tmp_base / "parquet_out"),
        )
        report = processor.run(input_path=csv_path, output_fmt="parquet")
        result = pd.read_parquet(report["output_path"])
        assert "prediction" in result.columns


# 7. Performance benchmarks (smoke tests)

class TestPerformance:
    def test_prediction_speed(self, predictor):
        """Batch prediction of 1 000 rows should complete in under 10 seconds."""
        import time

        df = generate_synthetic_data(n_samples=1000, random_state=7)
        start = time.time()
        predictor.predict_batch(df)
        elapsed = time.time() - start
        assert elapsed < 10, f"Prediction too slow: {elapsed:.2f}s"

    def test_trainer_speed(self, csv_path, tmp_base):
        """Full training pipeline on 1 000 rows should complete in under 60 seconds."""
        import time

        trainer = FraudDetectionTrainer(
            data_path=csv_path,
            model_dir=str(tmp_base / "perf_models"),
        )
        start = time.time()
        trainer.run_training_pipeline()
        elapsed = time.time() - start
        assert elapsed < 60, f"Training too slow: {elapsed:.2f}s"
```

**Test categories:**

| # | Category | Tests | Purpose |
|---|----------|-------|---------|
| 1 | Data Generation | 6 | Shape, columns, fraud ratio, nulls, class values, reproducibility |
| 2 | Model Training | 8 | Metrics returned, ROC-AUC/F1 bounds, file existence, JSON metadata |
| 3 | Single Prediction | 4 | Output keys, binary prediction, probability range, determinism |
| 4 | Batch Prediction | 5 | Columns added, row count preserved, value ranges, CSV round-trip |
| 5 | Batch Processor | 4 | Full pipeline, report keys, output file creation, missing model error |
| 6 | Output Format | 2 | CSV and Parquet readability |
| 7 | Performance | 2 | Prediction < 10 s, training < 60 s |

---

### `requirements.txt` — Python Dependencies

```
pyspark==3.5.0
kafka-python==2.0.2
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
matplotlib==3.5.3
seaborn==0.12.2
jupyter==1.0.0
joblib==1.4.2
pytest==7.1.3
pyarrow==14.0.1
```

| Package | Role |
|---------|------|
| `pyspark` | Spark Structured Streaming engine |
| `kafka-python` | Kafka producer client |
| `pandas` / `numpy` | Data manipulation and numerical operations |
| `scikit-learn` | Logistic Regression model and evaluation metrics |
| `matplotlib` / `seaborn` | Visualisations in Jupyter notebooks |
| `jupyter` | Interactive notebook server |
| `joblib` | Model serialisation / deserialisation |
| `pytest` | Test framework |
| `pyarrow` | Parquet file I/O |

---

### Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_exploratory_data_analysis.ipynb` | Loads the dataset, inspects shape and types, checks for missing values, visualises class distribution (pie and bar charts), plots amount distributions by class, analyses time-based patterns, and computes a correlation matrix for `V1`–`V10`, `Amount`, and `Class`. |
| `notebooks/02_model_training.ipynb` | Preprocesses data with `StandardScaler`, trains a `LogisticRegression(class_weight='balanced')` model, evaluates with classification report, confusion matrix heatmap, ROC curve, and precision-recall curve, then saves model and scaler with `joblib`. |

See [`notebooks/README.md`](notebooks/README.md) for setup instructions.

## License

MIT License — see [LICENSE](LICENSE) for details.
