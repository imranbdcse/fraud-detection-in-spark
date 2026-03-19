"""
Unit Tests: Spark Streaming Processor (src/streaming/stream_processor.py)
"""
import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.streaming.stream_processor import (
    parse_record,
    encode_transaction,
    MERCHANT_TYPE_MAP,
    TRANSACTION_TYPE_MAP,
    FEATURE_COLS,
)


SAMPLE_TRANSACTION = {
    "TransactionID": "txn-001",
    "UserID": "user_1234",
    "Timestamp": "2024-01-01T12:00:00Z",
    "TransactionAmount": 250.75,
    "IPAddress": "192.168.1.1",
    "Location": "New York",
    "MerchantType": "grocery",
    "CardLastFour": "4321",
    "TransactionType": "purchase",
    "DeviceID": "dev-abc",
    "FraudLabel": 0,
}


class TestParseRecord(unittest.TestCase):
    """Tests for parse_record."""

    def test_valid_json(self):
        raw = json.dumps(SAMPLE_TRANSACTION)
        result = parse_record(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["TransactionID"], "txn-001")

    def test_invalid_json_returns_none(self):
        result = parse_record("{not valid json}")
        self.assertIsNone(result)

    def test_none_input_returns_none(self):
        result = parse_record(None)
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        result = parse_record("")
        self.assertIsNone(result)

    def test_preserves_all_fields(self):
        raw = json.dumps(SAMPLE_TRANSACTION)
        result = parse_record(raw)
        for key in SAMPLE_TRANSACTION:
            self.assertIn(key, result)


class TestEncodeTransaction(unittest.TestCase):
    """Tests for encode_transaction."""

    def test_returns_row_for_valid_input(self):
        from pyspark.sql import Row
        row = encode_transaction(SAMPLE_TRANSACTION)
        self.assertIsNotNone(row)
        self.assertIsInstance(row, Row)

    def test_merchant_type_encoded(self):
        tx = dict(SAMPLE_TRANSACTION, MerchantType="grocery")
        row = encode_transaction(tx)
        self.assertEqual(row.MerchantTypeEncoded, float(MERCHANT_TYPE_MAP["grocery"]))

    def test_unknown_merchant_type_defaults_to_zero(self):
        tx = dict(SAMPLE_TRANSACTION, MerchantType="unknown_type")
        row = encode_transaction(tx)
        self.assertEqual(row.MerchantTypeEncoded, 0.0)

    def test_transaction_type_encoded(self):
        tx = dict(SAMPLE_TRANSACTION, TransactionType="withdrawal")
        row = encode_transaction(tx)
        self.assertEqual(row.TransactionTypeEncoded, float(TRANSACTION_TYPE_MAP["withdrawal"]))

    def test_transaction_amount_is_float(self):
        row = encode_transaction(SAMPLE_TRANSACTION)
        self.assertIsInstance(row.TransactionAmount, float)

    def test_fraud_label_preserved(self):
        tx_normal = dict(SAMPLE_TRANSACTION, FraudLabel=0)
        tx_fraud = dict(SAMPLE_TRANSACTION, FraudLabel=1)
        self.assertEqual(encode_transaction(tx_normal).FraudLabel, 0)
        self.assertEqual(encode_transaction(tx_fraud).FraudLabel, 1)

    def test_missing_optional_field_uses_default(self):
        tx = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "Location"}
        row = encode_transaction(tx)
        self.assertIsNotNone(row)
        self.assertEqual(row.Location, "")

    def test_feature_cols_all_present_in_row(self):
        row = encode_transaction(SAMPLE_TRANSACTION)
        row_dict = row.asDict()
        for col in FEATURE_COLS:
            self.assertIn(col, row_dict, f"Feature column '{col}' missing from Row")


class TestMappings(unittest.TestCase):
    """Sanity checks for encoding maps."""

    def test_merchant_type_map_non_empty(self):
        self.assertGreater(len(MERCHANT_TYPE_MAP), 0)

    def test_merchant_type_values_are_ints(self):
        for k, v in MERCHANT_TYPE_MAP.items():
            self.assertIsInstance(v, int, f"{k} value should be int")

    def test_transaction_type_map_non_empty(self):
        self.assertGreater(len(TRANSACTION_TYPE_MAP), 0)

    def test_transaction_type_values_are_ints(self):
        for k, v in TRANSACTION_TYPE_MAP.items():
            self.assertIsInstance(v, int, f"{k} value should be int")

    def test_feature_cols_contains_expected_columns(self):
        self.assertIn("TransactionAmount", FEATURE_COLS)
        self.assertIn("MerchantTypeEncoded", FEATURE_COLS)
        self.assertIn("TransactionTypeEncoded", FEATURE_COLS)


if __name__ == "__main__":
    unittest.main()
