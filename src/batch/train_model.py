"""
Fraud Detection Model Training Pipeline.

This script provides a complete, production-ready training pipeline for a
credit card fraud detection model using Logistic Regression.

Usage
-----
    python train_model.py
    python train_model.py --data-path ../../data/creditcard.csv
    python train_model.py --data-path ../../data/creditcard.csv --model-dir ../../models --test-size 0.2
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from *self.data_path*.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If required columns are missing.
        """
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
        """
        Split and scale features.

        Applies a stratified 80/20 train/test split and fits a
        ``StandardScaler`` on the training set.
        """
        if self.df is None:
            raise RuntimeError("Call load_data() before prepare_data().")

        logger.info("Preparing data (test_size=%.0f%%)", self.test_size * 100)
        X = self.df[self.FEATURE_COLUMNS].values
        y = self.df[self.TARGET_COLUMN].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info(
            "Train: %d samples | Test: %d samples",
            len(self.X_train),
            len(self.X_test),
        )

    def train_model(self) -> LogisticRegression:
        """
        Train a Logistic Regression model with balanced class weights.

        Returns
        -------
        LogisticRegression
            Fitted model.
        """
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
        """
        Evaluate the trained model with comprehensive metrics.

        Computes classification report, confusion matrix, ROC-AUC,
        F1 score, and 5-fold cross-validated ROC-AUC.

        Returns
        -------
        dict
            Dictionary containing all evaluation metrics.
        """
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
        """
        Save the trained model and scaler to *self.model_dir*.

        Two copies are saved:
        - ``fraud_model.pkl`` / ``scaler.pkl``  — production artefacts
        - ``fraud_model_<timestamp>.pkl`` / ``scaler_<timestamp>.pkl`` — versioned backups

        Args:
            metrics: Optional evaluation metrics to store alongside the model.

        Returns
        -------
        dict
            Paths to all saved artefacts.
        """
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
        """
        Execute the complete training pipeline.

        Steps:
        1. Load data
        2. Prepare (split + scale)
        3. Train
        4. Evaluate
        5. Save

        Returns
        -------
        dict
            Evaluation metrics and saved artefact paths.
        """
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

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

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


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

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
        "--test-size",
        type=float,
        default=0.2,
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
