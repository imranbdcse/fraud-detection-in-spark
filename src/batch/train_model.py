"""
FraudDetectionTrainer: Complete training pipeline for credit card fraud detection.

Usage:
    python train_model.py --data_path ../../data/creditcard.csv --model_dir ../../models --test_size 0.2
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """End-to-end trainer for the credit card fraud detection model."""

    def __init__(self, data_path: str, model_dir: str, test_size: float = 0.2):
        self.data_path = data_path
        self.model_dir = model_dir
        self.test_size = test_size
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

        os.makedirs(self.model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV."""
        logger.info("Loading data from %s", self.data_path)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        fraud_count = df["Class"].sum()
        logger.info(
            "Fraud samples: %d (%.2f%%)", fraud_count, fraud_count / len(df) * 100
        )
        return df

    def prepare_data(self, df: pd.DataFrame) -> None:
        """Split into train/test sets and apply feature scaling."""
        logger.info("Preparing data (test_size=%.2f, stratified split)…", self.test_size)
        feature_cols = [c for c in df.columns if c != "Class"]
        X = df[feature_cols].values
        y = df["Class"].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info(
            "Train: %d samples | Test: %d samples",
            len(self.X_train),
            len(self.X_test),
        )

    def train_model(self) -> None:
        """Train a Logistic Regression model with balanced class weights."""
        logger.info("Training Logistic Regression model…")
        start = time.time()
        self.model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
        )
        self.model.fit(self.X_train, self.y_train)
        elapsed = time.time() - start
        logger.info("Training completed in %.2f seconds.", elapsed)

    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model.

        Returns a dictionary with all key metrics including cross-validation scores.
        """
        logger.info("Evaluating model…")
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        f1 = f1_score(self.y_test, y_pred)

        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=5, scoring="f1"
        )

        logger.info("\n%s", classification_report(self.y_test, y_pred))
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("ROC-AUC Score : %.4f", roc_auc)
        logger.info("F1 Score      : %.4f", f1)
        logger.info(
            "Cross-Val F1  : %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
        )

        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc_auc": roc_auc,
            "f1_score": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        }

    def save_model(self, metrics: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Persist the model and scaler.

        Saves:
        - ``models/fraud_model.pkl`` – production model
        - ``models/fraud_model_YYYYMMDD_HHMMSS.pkl`` – versioned backup
        - ``models/scaler.pkl`` – feature scaler

        Returns the paths of the production model and scaler.
        """
        logger.info("Saving model artifacts to %s", self.model_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        prod_model_path = os.path.join(self.model_dir, "fraud_model.pkl")
        versioned_model_path = os.path.join(
            self.model_dir, f"fraud_model_{timestamp}.pkl"
        )
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")

        joblib.dump(self.model, prod_model_path)
        joblib.dump(self.model, versioned_model_path)
        joblib.dump(self.scaler, scaler_path)

        logger.info("Production model : %s", prod_model_path)
        logger.info("Versioned backup : %s", versioned_model_path)
        logger.info("Scaler           : %s", scaler_path)

        # Optionally save metrics alongside the versioned model
        if metrics:
            import json

            metrics_path = os.path.join(
                self.model_dir, f"metrics_{timestamp}.json"
            )
            with open(metrics_path, "w") as fh:
                json.dump(
                    {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in metrics.items()},
                    fh,
                    indent=2,
                )
            logger.info("Metrics          : %s", metrics_path)

        return prod_model_path, scaler_path

    def run_training_pipeline(self) -> Dict:
        """Execute the complete training pipeline end-to-end."""
        logger.info("=" * 60)
        logger.info("Starting Fraud Detection Training Pipeline")
        logger.info("=" * 60)

        df = self.load_data()
        self.prepare_data(df)
        self.train_model()
        metrics = self.evaluate_model()
        model_path, scaler_path = self.save_model(metrics)

        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info("  Model  : %s", model_path)
        logger.info("  Scaler : %s", scaler_path)
        logger.info("=" * 60)

        return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the credit card fraud detection model."
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(__file__), "../../data/creditcard.csv"),
        help="Path to the creditcard.csv dataset.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "../../models"),
        help="Directory where trained model artifacts will be saved.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = FraudDetectionTrainer(
        data_path=args.data_path,
        model_dir=args.model_dir,
        test_size=args.test_size,
    )
    trainer.run_training_pipeline()
