"""
Batch processor for historical credit card fraud detection.

Loads transaction data from various sources, applies the trained fraud-detection
model, generates reports, and saves results with proper partitioning.

Usage:
    python batch_processor.py --input_path ../../data/creditcard.csv \\
                              --output_dir ../../data/predictions \\
                              --model_dir ../../models
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process historical transaction data and apply fraud detection."""

    def __init__(self, model_dir: str, output_dir: str):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.predictor = None
        os.makedirs(output_dir, exist_ok=True)
        self._init_predictor()

    def _init_predictor(self) -> None:
        """Lazy-import FraudPredictor to avoid hard dependency at import time."""
        # Allow running from any working directory
        batch_dir = os.path.dirname(os.path.abspath(__file__))
        if batch_dir not in sys.path:
            sys.path.insert(0, batch_dir)
        from predict import FraudPredictor  # noqa: PLC0415

        self.predictor = FraudPredictor(model_dir=self.model_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load transaction data from a CSV file."""
        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load transaction data from a Parquet file."""
        logger.info("Loading Parquet: %s", path)
        df = pd.read_parquet(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    def load_jdbc(self, jdbc_url: str, table: str, user: str, password: str) -> pd.DataFrame:
        """Load transaction data from a JDBC source via SQLAlchemy."""
        try:
            from sqlalchemy import create_engine  # noqa: PLC0415

            engine = create_engine(jdbc_url)
            df = pd.read_sql_table(table, con=engine)
            logger.info("Loaded %d rows from %s:%s", len(df), jdbc_url, table)
            return df
        except ImportError:
            raise RuntimeError("sqlalchemy is required for JDBC data loading.")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fraud detection model to the input DataFrame."""
        logger.info("Running fraud detection on %d transactions…", len(df))
        results = self.predictor.predict_batch(df)
        fraud_count = results["prediction"].sum()
        logger.info(
            "Fraud detected: %d / %d (%.2f%%)",
            fraud_count,
            len(results),
            fraud_count / len(results) * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Output / reporting
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, partition_by: str = "date") -> str:
        """
        Save prediction results with partitioning.

        Args:
            df: DataFrame with prediction columns appended.
            partition_by: Partition strategy; currently 'date' is supported.

        Returns:
            Path to the saved output file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"predictions_{timestamp}.parquet")
        df.to_parquet(output_path, index=False)
        logger.info("Results saved to: %s", output_path)
        return output_path

    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate a text summary report of batch predictions.

        Args:
            df: DataFrame with 'prediction' and 'fraud_probability' columns.

        Returns:
            Report text as a string.
        """
        total = len(df)
        fraud = df["prediction"].sum()
        normal = total - fraud
        avg_prob = df["fraud_probability"].mean()
        high_risk = (df["fraud_probability"] >= 0.8).sum()

        lines = [
            "=" * 50,
            "Fraud Detection Batch Report",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 50,
            f"Total transactions  : {total:,}",
            f"Normal (0)          : {normal:,} ({normal / total * 100:.2f}%)",
            f"Fraud  (1)          : {fraud:,}  ({fraud / total * 100:.2f}%)",
            f"Avg fraud probability: {avg_prob:.4f}",
            f"High-risk (>=0.8)   : {high_risk:,}",
            "=" * 50,
        ]
        report = "\n".join(lines)
        logger.info("\n%s", report)
        return report

    # ------------------------------------------------------------------
    # Incremental processing
    # ------------------------------------------------------------------

    def process_incremental(self, input_dir: str, processed_log: str) -> None:
        """
        Process only new files in *input_dir* (incremental mode).

        Already-processed files are tracked in *processed_log*.
        """
        processed = set()
        if os.path.exists(processed_log):
            with open(processed_log) as fh:
                processed = {line.strip() for line in fh}

        new_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".csv") and os.path.join(input_dir, f) not in processed
        ]

        if not new_files:
            logger.info("No new files to process.")
            return

        logger.info("Found %d new file(s) to process.", len(new_files))
        for filepath in new_files:
            df = self.load_csv(filepath)
            results = self.process(df)
            self.save_results(results)
            self.generate_report(results)
            with open(processed_log, "a") as fh:
                fh.write(filepath + "\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-process historical transaction data for fraud detection."
    )
    parser.add_argument(
        "--input_path",
        default=os.path.join(os.path.dirname(__file__), "../../data/creditcard.csv"),
        help="Path to input CSV / Parquet file.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "../../data/predictions"),
        help="Directory to save prediction results.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "../../models"),
        help="Directory containing fraud_model.pkl and scaler.pkl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processor = BatchProcessor(model_dir=args.model_dir, output_dir=args.output_dir)

    if args.input_path.endswith(".parquet"):
        df = processor.load_parquet(args.input_path)
    else:
        df = processor.load_csv(args.input_path)

    results = processor.process(df)
    processor.save_results(results)
    processor.generate_report(results)
