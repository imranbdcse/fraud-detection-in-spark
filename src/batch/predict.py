"""
FraudPredictor: Load a saved model and run inference on new transactions.

Usage:
    python predict.py --model_dir ../../models --input_csv ../../data/sample.csv
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Load a persisted fraud-detection model and run predictions."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self._load_artifacts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load the production model and scaler from *model_dir*."""
        model_path = os.path.join(self.model_dir, "fraud_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Model loaded from  : %s", model_path)
        logger.info("Scaler loaded from : %s", scaler_path)

    def _preprocess(self, features: np.ndarray) -> np.ndarray:
        """Apply the saved scaler to raw feature array."""
        return self.scaler.transform(features)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, features: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        """
        Predict whether a single transaction is fraudulent.

        Args:
            features: 1-D array-like of length 30 (V1-V28, Amount, Time).

        Returns:
            Tuple of (prediction, fraud_probability) where prediction is 0 or 1.
        """
        arr = np.array(features, dtype=float).reshape(1, -1)
        arr_scaled = self._preprocess(arr)
        prediction = int(self.model.predict(arr_scaled)[0])
        probability = float(self.model.predict_proba(arr_scaled)[0][1])
        return prediction, probability

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run fraud detection on a batch of transactions.

        Args:
            df: DataFrame whose columns match the training feature set
                (Class column is ignored if present).

        Returns:
            The input DataFrame with two additional columns:
            ``prediction`` (0/1) and ``fraud_probability``.
        """
        feature_cols = [c for c in df.columns if c != "Class"]
        X = df[feature_cols].values
        X_scaled = self._preprocess(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        result = df.copy()
        result["prediction"] = predictions
        result["fraud_probability"] = probabilities
        return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fraud detection predictions using a saved model."
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "../../models"),
        help="Directory containing fraud_model.pkl and scaler.pkl.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Path to a CSV file to run batch predictions on (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    predictor = FraudPredictor(model_dir=args.model_dir)

    # Example: single prediction with random data
    logger.info("\n--- Single prediction example ---")
    sample_features = np.random.randn(30).tolist()
    pred, prob = predictor.predict_single(sample_features)
    logger.info("Prediction: %d  |  Fraud probability: %.4f", pred, prob)

    # Batch prediction from CSV if provided
    if args.input_csv:
        logger.info("\n--- Batch prediction from %s ---", args.input_csv)
        df = pd.read_csv(args.input_csv)
        results = predictor.predict_batch(df)
        fraud_detected = results["prediction"].sum()
        logger.info(
            "Processed %d transactions | Fraud detected: %d", len(results), fraud_detected
        )
        output_path = args.input_csv.replace(".csv", "_predictions.csv")
        results.to_csv(output_path, index=False)
        logger.info("Results saved to: %s", output_path)
