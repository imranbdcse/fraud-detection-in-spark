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

Run
---
    pytest tests/test_end_to_end.py -v
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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

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
        data_path=csv_path,
        model_dir=model_dir,
        test_size=0.2,
    )
    return trainer.run_training_pipeline()


@pytest.fixture(scope="session")
def predictor(trained_artefacts):
    """Return an initialised FraudPredictor backed by the session-level model."""
    return FraudPredictor(
        model_path=trained_artefacts["paths"]["model"],
        scaler_path=trained_artefacts["paths"]["scaler"],
    )


# --------------------------------------------------------------------------- #
# 1. Data generation
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 2. Model training
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 3. Single prediction
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 4. Batch prediction
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 5. Batch processor
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 6. Data flow / output format
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 7. Performance benchmarks (smoke tests)
# --------------------------------------------------------------------------- #

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
