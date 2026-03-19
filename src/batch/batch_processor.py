"""
Batch processor for historical credit card fraud detection.

Loads transaction data from CSV or Parquet, applies the trained fraud
detection model, and writes results with proper partitioning.

Usage
-----
    python batch_processor.py
    python batch_processor.py --input ../../data/creditcard.csv --format csv
    python batch_processor.py --input /data/transactions.parquet --format parquet \\
        --output /results/predictions --partition-by date
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

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_data(self, path: str, fmt: str = "csv") -> pd.DataFrame:
        """
        Load transaction data from *path*.

        Parameters
        ----------
        path : str
            Path to the input file or directory.
        fmt : str
            Source format: ``"csv"`` or ``"parquet"``.

        Returns
        -------
        pd.DataFrame
            Raw transaction data.
        """
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

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fraud detection model to *df*.

        Processes data in chunks to support large datasets without
        exhausting memory.

        Parameters
        ----------
        df : pd.DataFrame
            Input transactions (must contain all feature columns).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``prediction`` and ``fraud_probability`` columns.
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

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def save_results(
        self,
        results: pd.DataFrame,
        fmt: str = "csv",
        partition_by: str = None,
    ) -> str:
        """
        Save prediction results to *self.output_dir*.

        Parameters
        ----------
        results : pd.DataFrame
            Prediction results DataFrame.
        fmt : str
            Output format: ``"csv"`` or ``"parquet"``.
        partition_by : str, optional
            Column name to use for directory partitioning (Parquet only).

        Returns
        -------
        str
            Path to the output file or directory.
        """
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

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def generate_report(self, results: pd.DataFrame) -> dict:
        """
        Generate a summary report for the batch predictions.

        Parameters
        ----------
        results : pd.DataFrame
            Prediction results.

        Returns
        -------
        dict
            Summary statistics.
        """
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
            total,
            fraud_count,
            report["fraud_rate_pct"],
            len(high_risk),
            len(medium_risk),
        )
        return report

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def run(
        self,
        input_path: str,
        input_fmt: str = "csv",
        output_fmt: str = "csv",
        partition_by: str = None,
    ) -> dict:
        """
        Run the full batch processing pipeline.

        Parameters
        ----------
        input_path : str
            Path to input data.
        input_fmt : str
            Input format (``"csv"`` or ``"parquet"``).
        output_fmt : str
            Output format (``"csv"`` or ``"parquet"``).
        partition_by : str, optional
            Column for Parquet partitioning.

        Returns
        -------
        dict
            Batch report with summary statistics.
        """
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


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

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
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Rows per processing chunk."
    )
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
