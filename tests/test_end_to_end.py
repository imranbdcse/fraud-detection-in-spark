"""
End-to-end integration tests for the fraud detection pipeline.

These tests validate:
  - Synthetic data generation
  - Model training and evaluation
  - Batch prediction (single and batch)
  - Data flow correctness
  - Output format and accuracy
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Make sure src/batch is importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "batch"))
sys.path.insert(0, os.path.join(_ROOT, "data"))


class TestDataGeneration(unittest.TestCase):
    """Tests for generate_synthetic_data.py."""

    def test_default_generation(self):
        from generate_synthetic_data import generate_synthetic_data

        df = generate_synthetic_data(n_samples=1000, fraud_ratio=0.02)
        self.assertEqual(len(df), 1000)
        self.assertIn("Class", df.columns)
        self.assertEqual(set(df["Class"].unique()), {0, 1})

    def test_feature_columns_present(self):
        from generate_synthetic_data import generate_synthetic_data

        df = generate_synthetic_data(n_samples=500)
        expected = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
        for col in expected:
            self.assertIn(col, df.columns)

    def test_fraud_ratio(self):
        from generate_synthetic_data import generate_synthetic_data

        df = generate_synthetic_data(n_samples=10000, fraud_ratio=0.02)
        actual_ratio = df["Class"].sum() / len(df)
        self.assertAlmostEqual(actual_ratio, 0.02, delta=0.005)

    def test_amount_non_negative(self):
        from generate_synthetic_data import generate_synthetic_data

        df = generate_synthetic_data(n_samples=1000)
        self.assertTrue((df["Amount"] >= 0).all())


class TestModelTraining(unittest.TestCase):
    """Tests for train_model.py FraudDetectionTrainer."""

    @classmethod
    def setUpClass(cls):
        """Generate a small dataset and train a model once for all tests."""
        from generate_synthetic_data import generate_synthetic_data
        from train_model import FraudDetectionTrainer

        cls.tmp_dir = tempfile.mkdtemp()
        cls.data_path = os.path.join(cls.tmp_dir, "creditcard.csv")
        cls.model_dir = os.path.join(cls.tmp_dir, "models")

        df = generate_synthetic_data(n_samples=2000, fraud_ratio=0.05)
        df.to_csv(cls.data_path, index=False)

        cls.trainer = FraudDetectionTrainer(
            data_path=cls.data_path,
            model_dir=cls.model_dir,
            test_size=0.2,
        )
        cls.metrics = cls.trainer.run_training_pipeline()

    def test_model_files_created(self):
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "fraud_model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "scaler.pkl")))

    def test_metrics_keys_present(self):
        required_keys = {"roc_auc", "f1_score", "cv_f1_mean", "cv_f1_std", "confusion_matrix"}
        for key in required_keys:
            self.assertIn(key, self.metrics)

    def test_roc_auc_range(self):
        self.assertGreater(self.metrics["roc_auc"], 0.5)
        self.assertLessEqual(self.metrics["roc_auc"], 1.0)

    def test_f1_score_range(self):
        self.assertGreaterEqual(self.metrics["f1_score"], 0.0)
        self.assertLessEqual(self.metrics["f1_score"], 1.0)

    def test_confusion_matrix_shape(self):
        cm = self.metrics["confusion_matrix"]
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)


class TestFraudPredictor(unittest.TestCase):
    """Tests for predict.py FraudPredictor."""

    @classmethod
    def setUpClass(cls):
        """Re-use the trained model from TestModelTraining."""
        from generate_synthetic_data import generate_synthetic_data
        from train_model import FraudDetectionTrainer
        from predict import FraudPredictor

        cls.tmp_dir = tempfile.mkdtemp()
        data_path = os.path.join(cls.tmp_dir, "creditcard.csv")
        model_dir = os.path.join(cls.tmp_dir, "models")

        df = generate_synthetic_data(n_samples=2000, fraud_ratio=0.05)
        df.to_csv(data_path, index=False)

        trainer = FraudDetectionTrainer(
            data_path=data_path,
            model_dir=model_dir,
            test_size=0.2,
        )
        trainer.run_training_pipeline()
        cls.predictor = FraudPredictor(model_dir=model_dir)
        cls.sample_df = df.drop(columns=["Class"]).head(50)

    def test_single_prediction_output_types(self):
        features = np.random.randn(30).tolist()
        pred, prob = self.predictor.predict_single(features)
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_batch_prediction_columns(self):
        results = self.predictor.predict_batch(self.sample_df)
        self.assertIn("prediction", results.columns)
        self.assertIn("fraud_probability", results.columns)

    def test_batch_prediction_length(self):
        results = self.predictor.predict_batch(self.sample_df)
        self.assertEqual(len(results), len(self.sample_df))

    def test_batch_prediction_valid_values(self):
        results = self.predictor.predict_batch(self.sample_df)
        self.assertTrue(results["prediction"].isin([0, 1]).all())
        self.assertTrue((results["fraud_probability"] >= 0).all())
        self.assertTrue((results["fraud_probability"] <= 1).all())


class TestBatchProcessor(unittest.TestCase):
    """Tests for batch_processor.py BatchProcessor."""

    @classmethod
    def setUpClass(cls):
        from generate_synthetic_data import generate_synthetic_data
        from train_model import FraudDetectionTrainer

        cls.tmp_dir = tempfile.mkdtemp()
        cls.data_path = os.path.join(cls.tmp_dir, "creditcard.csv")
        cls.model_dir = os.path.join(cls.tmp_dir, "models")
        cls.output_dir = os.path.join(cls.tmp_dir, "output")

        df = generate_synthetic_data(n_samples=2000, fraud_ratio=0.05)
        df.to_csv(cls.data_path, index=False)

        trainer = FraudDetectionTrainer(
            data_path=cls.data_path,
            model_dir=cls.model_dir,
            test_size=0.2,
        )
        trainer.run_training_pipeline()

    def test_batch_processing_csv(self):
        from batch_processor import BatchProcessor

        processor = BatchProcessor(
            model_dir=self.model_dir, output_dir=self.output_dir
        )
        df = processor.load_csv(self.data_path)
        results = processor.process(df)
        self.assertIn("prediction", results.columns)
        self.assertIn("fraud_probability", results.columns)

    def test_report_generation(self):
        from batch_processor import BatchProcessor

        processor = BatchProcessor(
            model_dir=self.model_dir, output_dir=self.output_dir
        )
        df = processor.load_csv(self.data_path)
        results = processor.process(df)
        report = processor.generate_report(results)
        self.assertIn("Total transactions", report)
        self.assertIn("Fraud", report)

    def test_save_results_creates_file(self):
        from batch_processor import BatchProcessor

        processor = BatchProcessor(
            model_dir=self.model_dir, output_dir=self.output_dir
        )
        df = processor.load_csv(self.data_path)
        results = processor.process(df)
        output_path = processor.save_results(results)
        self.assertTrue(os.path.exists(output_path))


class TestDataFlowIntegration(unittest.TestCase):
    """Validate complete data flow from generation → training → prediction."""

    def test_full_pipeline(self):
        from generate_synthetic_data import generate_synthetic_data
        from train_model import FraudDetectionTrainer
        from predict import FraudPredictor

        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "creditcard.csv")
            model_dir = os.path.join(tmp, "models")

            # Step 1: Generate data
            df = generate_synthetic_data(n_samples=1500, fraud_ratio=0.04)
            df.to_csv(data_path, index=False)

            # Step 2: Train
            trainer = FraudDetectionTrainer(
                data_path=data_path, model_dir=model_dir, test_size=0.2
            )
            metrics = trainer.run_training_pipeline()
            self.assertGreater(metrics["roc_auc"], 0.5)

            # Step 3: Predict
            predictor = FraudPredictor(model_dir=model_dir)
            test_df = df.drop(columns=["Class"]).head(20)
            results = predictor.predict_batch(test_df)

            self.assertEqual(len(results), 20)
            self.assertIn("prediction", results.columns)
            self.assertIn("fraud_probability", results.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
