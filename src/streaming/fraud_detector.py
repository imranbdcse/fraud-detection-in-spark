"""
Real-time Spark Structured Streaming fraud detector.

Reads transaction JSON messages from the Kafka topic ``transactions``,
applies a trained fraud detection model, and writes results to:

  * Console (monitoring)
  * Parquet files (persistent storage)
  * Kafka topic ``fraud_alerts`` (downstream alerting)

Usage
-----
    spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.8 \\
        src/streaming/fraud_detector.py

Environment variables (overrides)
----------------------------------
    KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
    KAFKA_INPUT_TOPIC         default: transactions
    KAFKA_OUTPUT_TOPIC        default: fraud_alerts
    MODEL_PATH                default: models/fraud_model.pkl
    SCALER_PATH               default: models/scaler.pkl
    CHECKPOINT_DIR            default: /tmp/fraud_detector_checkpoint
    OUTPUT_DIR                default: /tmp/fraud_detector_output
"""

import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "fraud_alerts")
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/tmp/fraud_detector_checkpoint")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/fraud_detector_output")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load the trained model and scaler from disk."""
    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Loaded model from %s", model_path)
    logger.info("Loaded scaler from %s", scaler_path)
    return model, scaler


def build_schema():
    """Return the StructType schema for incoming transaction JSON messages."""
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    fields = [StructField("TransactionID", StringType(), True)]
    for col in FEATURE_COLUMNS:
        fields.append(StructField(col, DoubleType(), True))
    fields.append(StructField("Timestamp", DoubleType(), True))
    return StructType(fields)


def build_spark_session(app_name: str = "FraudDetector"):
    """Create and return a SparkSession with Kafka-compatible settings."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_kafka_stream(spark):
    """Return a streaming DataFrame reading from the transactions Kafka topic."""
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_transactions(raw_df, schema):
    """Parse the raw Kafka binary value column into structured columns."""
    from pyspark.sql.functions import col, from_json

    return (
        raw_df.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
        )
        .select("data.*", "kafka_timestamp")
    )


def apply_fraud_model(transactions_df, model, scaler):
    """
    Apply the fraud detection model to the streaming DataFrame.

    Uses a Pandas UDF (vectorised) so the model runs efficiently across
    Spark partitions.
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType
    import pandas as pd

    # Broadcast the model and scaler objects to all executors
    sc = transactions_df.sparkSession.sparkContext
    bc_model = sc.broadcast(model)
    bc_scaler = sc.broadcast(scaler)

    @pandas_udf(DoubleType())
    def predict_fraud_probability(*cols):
        X = pd.concat(list(cols), axis=1)
        X.columns = FEATURE_COLUMNS
        X_scaled = bc_scaler.value.transform(X.values)
        probs = bc_model.value.predict_proba(X_scaled)[:, 1]
        return pd.Series(probs)

    feature_cols = [transactions_df[c] for c in FEATURE_COLUMNS]
    return transactions_df.withColumn("fraud_probability", predict_fraud_probability(*feature_cols))


def add_prediction_flag(df):
    """Add a binary ``is_fraud`` column (1 when fraud_probability >= 0.5)."""
    from pyspark.sql.functions import col, when

    return df.withColumn(
        "is_fraud",
        when(col("fraud_probability") >= 0.5, 1).otherwise(0),
    )


def write_to_console(df, checkpoint_dir: str):
    """Write stream to console for monitoring (truncated output)."""
    return (
        df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "console"))
        .start()
    )


def write_to_parquet(df, output_dir: str, checkpoint_dir: str):
    """Write stream to Parquet files for persistent storage."""
    return (
        df.writeStream.outputMode("append")
        .format("parquet")
        .option("path", output_dir)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet"))
        .trigger(processingTime="30 seconds")
        .start()
    )


def write_fraud_alerts_to_kafka(df, checkpoint_dir: str):
    """Forward fraud alerts to the fraud_alerts Kafka topic as JSON."""
    from pyspark.sql.functions import col, struct, to_json

    alert_df = df.filter(col("is_fraud") == 1).select(
        to_json(
            struct(
                col("TransactionID"),
                col("fraud_probability"),
                col("kafka_timestamp"),
            )
        ).alias("value")
    )

    return (
        alert_df.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", KAFKA_OUTPUT_TOPIC)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka"))
        .start()
    )


def main():
    logger.info("Starting Fraud Detector streaming job …")
    logger.info("  Kafka bootstrap : %s", KAFKA_BOOTSTRAP_SERVERS)
    logger.info("  Input topic     : %s", KAFKA_INPUT_TOPIC)
    logger.info("  Output topic    : %s", KAFKA_OUTPUT_TOPIC)

    try:
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    except FileNotFoundError as exc:
        logger.error("%s — train the model first with src/batch/train_model.py", exc)
        sys.exit(1)

    spark = build_spark_session()
    schema = build_schema()

    raw_stream = read_kafka_stream(spark)
    transactions = parse_transactions(raw_stream, schema)
    scored = apply_fraud_model(transactions, model, scaler)
    final = add_prediction_flag(scored)

    queries = [
        write_to_console(final, CHECKPOINT_DIR),
        write_to_parquet(final, OUTPUT_DIR, CHECKPOINT_DIR),
        write_fraud_alerts_to_kafka(final, CHECKPOINT_DIR),
    ]

    logger.info("Streaming queries started. Awaiting termination …")
    for q in queries:
        q.awaitTermination()


if __name__ == "__main__":
    main()
