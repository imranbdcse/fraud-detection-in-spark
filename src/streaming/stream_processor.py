"""
Spark Structured Streaming Processor: Real-Time Fraud Detection
===============================================================
Consumes credit card transaction events from a Kafka topic using
Spark Structured Streaming (Spark 3.x, spark-sql-kafka-0-10 connector),
preprocesses each micro-batch, applies a pretrained MLlib model to
classify transactions as fraudulent or legitimate, and writes flagged
events to a JSON output path.

NOTE: The legacy DStream + spark-streaming-kafka-0-8 API was removed in
Spark 3.0. This module uses the Structured Streaming API (readStream /
writeStream.foreachBatch) which is the recommended approach from Spark 2.0+
and is fully supported in Spark 3.x.

Usage (via spark-submit):
    spark-submit \\
        --master local[*] \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \\
        src/streaming/stream_processor.py \\
        --topic transactions \\
        --broker localhost:9092 \\
        --model-path models/logistic_regression_fraud_model \\
        --output-path output/fraud_alerts \\
        --trigger-seconds 5
"""

import argparse
import json
import logging

from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType,
)
from pyspark.ml.feature import VectorAssembler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — used by from_json to parse the Kafka value column
# ---------------------------------------------------------------------------
TRANSACTION_SCHEMA = StructType([
    StructField("TransactionID",     StringType(),  True),
    StructField("UserID",            StringType(),  True),
    StructField("Timestamp",         StringType(),  True),
    StructField("TransactionAmount", DoubleType(),  True),
    StructField("IPAddress",         StringType(),  True),
    StructField("Location",          StringType(),  True),
    StructField("MerchantType",      StringType(),  True),
    StructField("CardLastFour",      StringType(),  True),
    StructField("TransactionType",   StringType(),  True),
    StructField("DeviceID",          StringType(),  True),
    StructField("FraudLabel",        IntegerType(), True),
])

# Feature columns used during model training
FEATURE_COLS = ["TransactionAmount", "MerchantTypeEncoded", "TransactionTypeEncoded"]

# Merchant type mapping (must match training encoding)
MERCHANT_TYPE_MAP = {
    "grocery": 0, "online_retail": 1, "restaurant": 2,
    "electronics": 3, "travel": 4, "gas_station": 5,
    "pharmacy": 6, "clothing": 7, "entertainment": 8,
}

# Transaction type mapping
TRANSACTION_TYPE_MAP = {"purchase": 0, "withdrawal": 1, "refund": 2}


# ---------------------------------------------------------------------------
# Parsing helpers (also used in unit tests)
# ---------------------------------------------------------------------------
def parse_record(raw: str):
    """Parse a raw JSON string into a transaction dict. Returns None on error."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse record: %s — %s", raw, exc)
        return None


def encode_transaction(record: dict):
    """
    Apply label encoding to categorical fields and return a Spark Row.
    Returns None if any required field is missing.
    """
    try:
        return Row(
            TransactionID=record.get("TransactionID", ""),
            UserID=record.get("UserID", ""),
            Timestamp=record.get("Timestamp", ""),
            TransactionAmount=float(record.get("TransactionAmount", 0.0)),
            IPAddress=record.get("IPAddress", ""),
            Location=record.get("Location", ""),
            MerchantType=record.get("MerchantType", "grocery"),
            MerchantTypeEncoded=float(
                MERCHANT_TYPE_MAP.get(record.get("MerchantType", ""), 0)
            ),
            CardLastFour=record.get("CardLastFour", ""),
            TransactionType=record.get("TransactionType", "purchase"),
            TransactionTypeEncoded=float(
                TRANSACTION_TYPE_MAP.get(record.get("TransactionType", ""), 0)
            ),
            DeviceID=record.get("DeviceID", ""),
            FraudLabel=int(record.get("FraudLabel", 0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Failed to encode record: %s — %s", record, exc)
        return None


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing (used inside foreachBatch)
# ---------------------------------------------------------------------------
def preprocess_df(df: DataFrame) -> DataFrame:
    """
    Apply integer encoding to categorical columns using Spark SQL map literals.

    Args:
        df: DataFrame with raw transaction columns.

    Returns:
        DataFrame with additional encoded columns:
        ``MerchantTypeEncoded`` and ``TransactionTypeEncoded``.
    """
    merchant_map = F.create_map(
        *[item for pair in [(F.lit(k), F.lit(v)) for k, v in MERCHANT_TYPE_MAP.items()]
          for item in pair]
    )
    txn_type_map = F.create_map(
        *[item for pair in [(F.lit(k), F.lit(v)) for k, v in TRANSACTION_TYPE_MAP.items()]
          for item in pair]
    )

    return (
        df
        .withColumn(
            "MerchantTypeEncoded",
            F.coalesce(merchant_map[F.col("MerchantType")], F.lit(0)).cast("double"),
        )
        .withColumn(
            "TransactionTypeEncoded",
            F.coalesce(txn_type_map[F.col("TransactionType")], F.lit(0)).cast("double"),
        )
        .withColumn("TransactionAmount", F.col("TransactionAmount").cast("double"))
    )


# ---------------------------------------------------------------------------
# Batch processor (called per micro-batch by foreachBatch)
# ---------------------------------------------------------------------------
def make_batch_processor(model, assembler: VectorAssembler, output_path: str):
    """
    Return a foreachBatch function that runs fraud inference on each micro-batch.

    Args:
        model:       Loaded Spark ML model.
        assembler:   Fitted VectorAssembler for feature construction.
        output_path: Directory path to append fraud alerts as JSON.

    Returns:
        A callable ``(batch_df, epoch_id) -> None`` suitable for
        ``writeStream.foreachBatch()``.
    """
    def process_batch(batch_df: DataFrame, epoch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            logger.info("Epoch %d: empty batch, skipping.", epoch_id)
            return

        encoded = preprocess_df(batch_df)
        feature_df = assembler.transform(encoded)
        predictions = model.transform(feature_df)

        fraud_df = predictions.filter(F.col("prediction") == 1.0)
        fraud_count = fraud_df.count()

        if fraud_count > 0:
            logger.warning(
                "Epoch %d: Detected %d fraudulent transaction(s).",
                epoch_id, fraud_count,
            )
            fraud_df.select(
                "TransactionID", "UserID", "Timestamp",
                "TransactionAmount", "Location", "IPAddress",
                "MerchantType", "prediction", "probability",
            ).write.mode("append").json(output_path)
        else:
            logger.info(
                "Epoch %d: No fraud detected (%d transaction(s) processed).",
                epoch_id, batch_df.count(),
            )

    return process_batch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_streaming_query(
    topic: str,
    broker: str,
    model_path: str,
    output_path: str,
    checkpoint_dir: str,
    trigger_seconds: int,
):
    """
    Build and start a Spark Structured Streaming query that reads from Kafka,
    parses JSON, assembles feature vectors, runs fraud inference, and appends
    flagged transactions to ``output_path``.

    Args:
        topic:           Kafka topic name.
        broker:          Kafka bootstrap server address.
        model_path:      Path to the saved Spark ML model.
        output_path:     Directory to write fraud alert JSON files.
        checkpoint_dir:  Checkpoint directory for fault tolerance.
        trigger_seconds: Processing trigger interval in seconds.

    Returns:
        An active ``StreamingQuery`` object.
    """
    from pyspark.ml.classification import LogisticRegressionModel  # noqa: PLC0415

    spark = (
        SparkSession.builder
        .appName("CreditCardFraudDetection")
        .config("spark.sql.streaming.stopTimeout", "10000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Loading model from %s", model_path)
    model = LogisticRegressionModel.load(model_path)
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

    # Read raw bytes from Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", broker)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse the JSON value column into transaction fields
    parsed_stream = raw_stream.select(
        F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("data")
    ).select("data.*")

    batch_fn = make_batch_processor(model, assembler, output_path)

    query = (
        parsed_stream.writeStream
        .foreachBatch(batch_fn)
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )

    logger.info(
        "Streaming query started — topic='%s', trigger=%ds, output='%s'",
        topic, trigger_seconds, output_path,
    )
    return query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spark Structured Streaming Fraud Detector (Spark 3.x)"
    )
    parser.add_argument("--topic",           default="transactions",   help="Kafka topic")
    parser.add_argument("--broker",          default="localhost:9092", help="Kafka broker")
    parser.add_argument("--model-path",      default="models/logistic_regression_fraud_model",
                        help="Path to saved Spark ML model")
    parser.add_argument("--output-path",     default="output/fraud_alerts",
                        help="Directory to write fraud alert JSONs")
    parser.add_argument("--trigger-seconds", type=int, default=5,
                        help="Processing trigger interval in seconds (default: 5)")
    parser.add_argument("--checkpoint-dir",  default="checkpoints",
                        help="Spark Streaming checkpoint directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    query = build_streaming_query(
        topic=args.topic,
        broker=args.broker,
        model_path=args.model_path,
        output_path=args.output_path,
        checkpoint_dir=args.checkpoint_dir,
        trigger_seconds=args.trigger_seconds,
    )
    query.awaitTermination()

