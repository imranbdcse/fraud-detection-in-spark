"""
Fraud Detection Predictor.

Load a saved model and scaler to make single or batch predictions.

Usage
-----
    python predict.py                                      # uses default paths
    python predict.py --model ../../models/fraud_model.pkl --scaler ../../models/scaler.pkl
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
        Path to the serialised ``LogisticRegression`` model (joblib).
    scaler_path : str
        Path to the serialised ``StandardScaler`` (joblib).
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #

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
            ``{"prediction": int, "fraud_probability": float}``
        """
        row = pd.DataFrame([features])[FEATURE_COLUMNS]
        scaled = self.scaler.transform(row)
        prediction = int(self.model.predict(scaled)[0])
        probability = float(self.model.predict_proba(scaled)[0][1])
        return {"prediction": prediction, "fraud_probability": round(probability, 6)}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for a DataFrame of transactions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the 30 feature columns.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with two new columns:
            ``prediction`` and ``fraud_probability``.
        """
        result = df.copy()
        X = result[FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X)
        result["prediction"] = self.model.predict(X_scaled).astype(int)
        result["fraud_probability"] = self.model.predict_proba(X_scaled)[:, 1].round(6)
        return result

    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Load a CSV file and run batch predictions.

        Parameters
        ----------
        csv_path : str
            Path to the input CSV file.
        output_path : str, optional
            If provided, save predictions to this CSV path.

        Returns
        -------
        pd.DataFrame
            Predictions DataFrame.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s", len(df), csv_path)

        predictions = self.predict_batch(df)
        fraud_count = predictions["prediction"].sum()
        logger.info(
            "Predicted %d fraud transactions out of %d (%.2f%%)",
            fraud_count,
            len(predictions),
            fraud_count / len(predictions) * 100,
        )

        if output_path:
            predictions.to_csv(output_path, index=False)
            logger.info("Predictions saved to %s", output_path)

        return predictions


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

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

    # ------------------------------------------------------------------ #
    # Example usage with a single synthetic transaction
    # ------------------------------------------------------------------ #
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
