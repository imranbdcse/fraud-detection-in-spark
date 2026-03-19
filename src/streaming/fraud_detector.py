"""
Real-time Spark Structured Streaming fraud detector.

Reads transaction JSON messages from Kafka topic "transactions",
applies a trained ML model, and writes alerts to:
  - Console  (monitoring)
  - Parquet  (persistent storage)
  - Kafka topic "fraud_alerts"  (downstream systems)

Usage:
    spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \\
        src/streaming/fraud_detector.py \\
        --model_dir models/ \\
        --checkpoint_dir /tmp/fraud_checkpoint
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "FraudDetection"):
    """Create and return a configured SparkSession."""
    from pyspark.sql import SparkSession  # noqa: PLC0415

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def build_transaction_schema():
    """Return the StructType schema that matches the Kafka producer output."""
    from pyspark.sql.types import (  # noqa: PLC0415
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    v_fields = [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
    return StructType(
        v_fields
        + [
            StructField("Amount", DoubleType(), True),
            StructField("Time", LongType(), True),
            StructField("TransactionID", StringType(), True),
            StructField("UserID", StringType(), True),
        ]
    )


def load_model_broadcast(spark, model_dir: str):
    """
    Load the trained model and scaler, broadcasting them to all executors.

    Returns (broadcast_model, broadcast_scaler).
    """
    import joblib  # noqa: PLC0415

    model = joblib.load(os.path.join(model_dir, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    bc_model = spark.sparkContext.broadcast(model)
    bc_scaler = spark.sparkContext.broadcast(scaler)
    logger.info("Model and scaler broadcast to all executors.")
    return bc_model, bc_scaler


def make_predict_udf(bc_model, bc_scaler):
    """Create a PySpark UDF that returns the fraud probability for a row."""
    from pyspark.sql.functions import udf  # noqa: PLC0415
    from pyspark.sql.types import DoubleType  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def predict_fraud(*features):
        arr = np.array(features, dtype=float).reshape(1, -1)
        arr_scaled = bc_scaler.value.transform(arr)
        prob = float(bc_model.value.predict_proba(arr_scaled)[0][1])
        return prob

    return udf(predict_fraud, DoubleType())


def run_streaming(
    kafka_bootstrap: str,
    input_topic: str,
    output_topic: str,
    model_dir: str,
    checkpoint_dir: str,
    output_path: str,
):
    """Start the Spark Structured Streaming fraud detection job."""
    from pyspark.sql.functions import col, from_json, to_json, struct, when  # noqa: PLC0415

    spark = get_spark_session()
    schema = build_transaction_schema()
    bc_model, bc_scaler = load_model_broadcast(spark, model_dir)
    predict_udf = make_predict_udf(bc_model, bc_scaler)

    # Read from Kafka
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", input_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    # Parse JSON payload
    transactions = raw_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    # Feature columns used during training
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

    # Apply fraud detection UDF
    transactions_with_pred = transactions.withColumn(
        "fraud_probability", predict_udf(*[col(c) for c in feature_cols])
    ).withColumn(
        "is_fraud", when(col("fraud_probability") >= 0.5, 1).otherwise(0)
    )

    # Sink 1: Console (monitoring)
    console_query = (
        transactions_with_pred.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoint_dir, "console"))
        .start()
    )

    # Sink 2: Parquet files (storage)
    parquet_query = (
        transactions_with_pred.writeStream.outputMode("append")
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet"))
        .start()
    )

    # Sink 3: Kafka "fraud_alerts" topic
    kafka_output = transactions_with_pred.select(
        to_json(struct([col(c) for c in transactions_with_pred.columns])).alias("value")
    )
    kafka_query = (
        kafka_output.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("topic", output_topic)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka"))
        .start()
    )

    logger.info("Streaming queries started. Awaiting termination…")
    spark.streams.awaitAnyTermination()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Spark Streaming fraud detector."
    )
    parser.add_argument("--kafka_bootstrap", default="localhost:9092")
    parser.add_argument("--input_topic", default="transactions")
    parser.add_argument("--output_topic", default="fraud_alerts")
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "../../models"),
    )
    parser.add_argument("--checkpoint_dir", default="/tmp/fraud_checkpoint")
    parser.add_argument("--output_path", default="/tmp/fraud_predictions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_streaming(
        kafka_bootstrap=args.kafka_bootstrap,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
    )
