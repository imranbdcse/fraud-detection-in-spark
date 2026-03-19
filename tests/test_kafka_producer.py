"""
Unit Tests: Kafka Producer (scripts/kafka_producer.py)
"""
import json
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.kafka_producer import generate_transaction, parse_args


class TestGenerateTransaction(unittest.TestCase):
    """Tests for the generate_transaction function."""

    def test_returns_dict(self):
        tx = generate_transaction()
        self.assertIsInstance(tx, dict)

    def test_required_fields_present(self):
        required_fields = [
            "TransactionID", "UserID", "Timestamp", "TransactionAmount",
            "IPAddress", "Location", "MerchantType", "CardLastFour",
            "TransactionType", "DeviceID", "FraudLabel",
        ]
        tx = generate_transaction()
        for field in required_fields:
            self.assertIn(field, tx, f"Missing field: {field}")

    def test_fraud_label_is_binary(self):
        for _ in range(20):
            tx = generate_transaction(fraud_ratio=0.5)
            self.assertIn(tx["FraudLabel"], [0, 1])

    def test_transaction_amount_positive(self):
        for _ in range(20):
            tx = generate_transaction()
            self.assertGreater(tx["TransactionAmount"], 0)

    def test_fraud_ratio_zero_produces_no_fraud(self):
        """With fraud_ratio=0, no transaction should be labelled as fraud."""
        for _ in range(50):
            tx = generate_transaction(fraud_ratio=0.0)
            self.assertEqual(tx["FraudLabel"], 0)

    def test_fraud_ratio_one_always_fraud(self):
        """With fraud_ratio=1, every transaction should be labelled as fraud."""
        for _ in range(20):
            tx = generate_transaction(fraud_ratio=1.0)
            self.assertEqual(tx["FraudLabel"], 1)

    def test_transaction_id_is_unique(self):
        ids = {generate_transaction()["TransactionID"] for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_merchant_type_valid(self):
        valid_types = {
            "grocery", "online_retail", "restaurant", "electronics",
            "travel", "gas_station", "pharmacy", "clothing", "entertainment",
        }
        for _ in range(20):
            tx = generate_transaction()
            self.assertIn(tx["MerchantType"], valid_types)

    def test_serialisable_to_json(self):
        tx = generate_transaction()
        try:
            json.dumps(tx)
        except (TypeError, ValueError) as exc:
            self.fail(f"Transaction is not JSON-serialisable: {exc}")


class TestParseArgs(unittest.TestCase):
    """Tests for argument parsing."""

    def test_defaults(self):
        with patch("sys.argv", ["kafka_producer.py"]):
            args = parse_args()
        self.assertEqual(args.topic, "transactions")
        self.assertEqual(args.broker, "localhost:9092")
        self.assertAlmostEqual(args.rate, 1.0)
        self.assertAlmostEqual(args.fraud_ratio, 0.05)
        self.assertEqual(args.max_messages, 0)

    def test_custom_args(self):
        with patch("sys.argv", [
            "kafka_producer.py",
            "--topic", "my_topic",
            "--broker", "kafka:9093",
            "--rate", "5.0",
            "--fraud-ratio", "0.1",
            "--max-messages", "100",
        ]):
            args = parse_args()
        self.assertEqual(args.topic, "my_topic")
        self.assertEqual(args.broker, "kafka:9093")
        self.assertAlmostEqual(args.rate, 5.0)
        self.assertAlmostEqual(args.fraud_ratio, 0.1)
        self.assertEqual(args.max_messages, 100)


class TestRunProducer(unittest.TestCase):
    """Tests for run_producer function."""

    @patch("scripts.kafka_producer.KafkaProducer")
    def test_sends_max_messages(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer

        from scripts.kafka_producer import run_producer
        run_producer(
            topic="transactions",
            broker="localhost:9092",
            rate=1000.0,
            fraud_ratio=0.0,
            max_messages=5,
        )

        self.assertEqual(mock_producer.send.call_count, 5)
        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()

    @patch("scripts.kafka_producer.KafkaProducer")
    def test_sends_to_correct_topic(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer

        from scripts.kafka_producer import run_producer
        run_producer(
            topic="test_topic",
            broker="localhost:9092",
            rate=1000.0,
            fraud_ratio=0.0,
            max_messages=1,
        )

        call_kwargs = mock_producer.send.call_args
        self.assertEqual(call_kwargs[0][0], "test_topic")


if __name__ == "__main__":
    unittest.main()
