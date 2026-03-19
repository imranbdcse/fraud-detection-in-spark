"""
Spark Streaming Processor: Real-Time Fraud Detection
=====================================================
Consumes credit card transaction events from a Kafka topic using
Spark Streaming (DStream API, Spark 2.0), preprocesses each micro-batch,
applies a pretrained MLlib model to classify transactions as fraudulent
or legitimate, and writes flagged events to a JSON output file.

Usage (via spark-submit):
    spark-submit \\
        --master local[*] \\
        --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 \\
        src/streaming/stream_processor.py \\
        --topic transactions \\
        --broker localhost:9092 \\
        --zk-quorum localhost:2181 \\
        --model-path models/logistic_regression_fraud_model \\
        --output-path output/fraud_alerts \\
        --batch-interval 5
"""

import argparse
import json
import logging

from pyspark.sql import SparkSession, Row
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
# Schema
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
# Parsing helpers
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
# Micro-batch processing
# ---------------------------------------------------------------------------
def process_batch(rdd, spark: SparkSession, model, output_path: str) -> None:
    """
    Process a single Spark Streaming micro-batch RDD:
      1. Parse and encode transactions.
      2. Assemble feature vectors.
      3. Run the MLlib model for inference.
      4. Persist fraudulent predictions.

    Args:
        rdd:         RDD of (key, value) tuples from Kafka.
        spark:       Active SparkSession.
        model:       Loaded Spark ML model (LogisticRegressionModel or PipelineModel).
        output_path: Directory path to write fraud alerts as JSON.
    """
    if rdd.isEmpty():
        return

    # Parse raw Kafka messages
    raw_records = rdd.map(lambda kv: kv[1])
    parsed = raw_records.map(parse_record).filter(lambda r: r is not None)
    encoded = parsed.map(encode_transaction).filter(lambda r: r is not None)

    if encoded.isEmpty():
        return

    # Convert to DataFrame
    df = spark.createDataFrame(encoded)

    # Assemble feature vector
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    feature_df = assembler.transform(df)

    # Run inference
    predictions = model.transform(feature_df)

    # Filter and persist fraudulent predictions
    fraud_df = predictions.filter(predictions["prediction"] == 1.0)
    fraud_count = fraud_df.count()

    if fraud_count > 0:
        logger.warning("Detected %d fraudulent transaction(s) in this batch.", fraud_count)
        fraud_df.select(
            "TransactionID", "UserID", "Timestamp",
            "TransactionAmount", "Location", "IPAddress",
            "MerchantType", "prediction", "probability",
        ).write.mode("append").json(output_path)
    else:
        logger.info("No fraud detected in this batch (%d transaction(s) processed).", df.count())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_streaming_context(
    topic: str,
    broker: str,
    zk_quorum: str,
    model_path: str,
    output_path: str,
    batch_interval: int,
    checkpoint_dir: str,
):
    """Build and configure the Spark StreamingContext."""
    # Lazy imports — these require Spark + Kafka JARs at runtime
    from pyspark.streaming import StreamingContext  # noqa: PLC0415
    from pyspark.streaming.kafka import KafkaUtils  # noqa: PLC0415
    from pyspark.ml.classification import LogisticRegressionModel  # noqa: PLC0415

    spark = (
        SparkSession.builder
        .appName("CreditCardFraudDetection")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Loading model from %s", model_path)
    model = LogisticRegressionModel.load(model_path)

    ssc = StreamingContext(spark.sparkContext, batchDuration=batch_interval)
    ssc.checkpoint(checkpoint_dir)

    kafka_stream = KafkaUtils.createStream(
        ssc,
        zk_quorum,
        "fraud-detection-consumer-group",
        {topic: 1},
    )

    kafka_stream.foreachRDD(
        lambda rdd: process_batch(rdd, spark, model, output_path)
    )

    return ssc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark Streaming Fraud Detector")
    parser.add_argument("--topic",          default="transactions",      help="Kafka topic")
    parser.add_argument("--broker",         default="localhost:9092",    help="Kafka broker")
    parser.add_argument("--zk-quorum",      default="localhost:2181",    help="ZooKeeper quorum")
    parser.add_argument("--model-path",     default="models/logistic_regression_fraud_model",
                        help="Path to saved Spark ML model")
    parser.add_argument("--output-path",    default="output/fraud_alerts",
                        help="Directory to write fraud alert JSONs")
    parser.add_argument("--batch-interval", type=int, default=5,
                        help="Streaming batch interval in seconds (default: 5)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Spark Streaming checkpoint directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ssc = build_streaming_context(
        topic=args.topic,
        broker=args.broker,
        zk_quorum=args.zk_quorum,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_interval=args.batch_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    ssc.start()
    ssc.awaitTermination()
