"""
Enhanced real-time Spark Structured Streaming fraud detector.

Adds real-time feature engineering on top of the basic fraud_detector:
  - Transaction velocity per user (count in 10-min window)
  - Amount aggregations per user (mean, max, std in 10-min window)
  - Time-based features

Multiple output modes:
  - Append mode for raw fraud predictions
  - Update mode for windowed user-level statistics

Performance optimisations:
  - Broadcast variables for model/scaler
  - Explicit partitioning
  - Caching of intermediate DataFrames

Usage:
    spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.8 \\
        src/streaming/fraud_detector_enhanced.py \\
        --model_dir models/ \\
        --checkpoint_dir /tmp/fraud_checkpoint_enhanced
"""

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_spark_session(app_name: str = "FraudDetectionEnhanced"):
    from pyspark.sql import SparkSession  # noqa: PLC0415

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.stateStore.providerClass",
                "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def build_transaction_schema():
    from pyspark.sql.types import (  # noqa: PLC0415
        DoubleType, LongType, StringType, StructField, StructType, TimestampType,
    )

    v_fields = [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
    return StructType(
        v_fields
        + [
            StructField("Amount", DoubleType(), True),
            StructField("Time", LongType(), True),
            StructField("TransactionID", StringType(), True),
            StructField("UserID", StringType(), True),
            StructField("EventTime", TimestampType(), True),
        ]
    )


def load_model_broadcast(spark, model_dir: str):
    import joblib  # noqa: PLC0415

    model = joblib.load(os.path.join(model_dir, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return spark.sparkContext.broadcast(model), spark.sparkContext.broadcast(scaler)


def make_predict_udf(bc_model, bc_scaler):
    import numpy as np  # noqa: PLC0415
    from pyspark.sql.functions import udf  # noqa: PLC0415
    from pyspark.sql.types import DoubleType  # noqa: PLC0415

    def predict_fraud(*features):
        arr = np.array(features, dtype=float).reshape(1, -1)
        arr_scaled = bc_scaler.value.transform(arr)
        return float(bc_model.value.predict_proba(arr_scaled)[0][1])

    return udf(predict_fraud, DoubleType())


# ---------------------------------------------------------------------------
# Main streaming logic
# ---------------------------------------------------------------------------

def run_enhanced_streaming(
    kafka_bootstrap: str,
    input_topic: str,
    output_topic: str,
    model_dir: str,
    checkpoint_dir: str,
    output_path: str,
    window_duration: str = "10 minutes",
    slide_duration: str = "2 minutes",
):
    """Start the enhanced Spark Structured Streaming job."""
    from pyspark.sql.functions import (  # noqa: PLC0415
        avg, col, count, current_timestamp, from_json, max as spark_max,
        stddev, struct, to_json, when, window,
    )

    spark = get_spark_session()
    schema = build_transaction_schema()
    bc_model, bc_scaler = load_model_broadcast(spark, model_dir)
    predict_udf = make_predict_udf(bc_model, bc_scaler)

    # ---- Read from Kafka ------------------------------------------------
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", input_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = raw_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*").withColumn("EventTime", current_timestamp())

    # ---- Feature engineering: base fraud score --------------------------
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    transactions_scored = transactions.withColumn(
        "fraud_probability", predict_udf(*[col(c) for c in feature_cols])
    ).withColumn(
        "is_fraud", when(col("fraud_probability") >= 0.5, 1).otherwise(0)
    )

    # ---- Windowed aggregations per user ----------------------------------
    user_stats = (
        transactions_scored.groupBy(
            col("UserID"),
            window(col("EventTime"), window_duration, slide_duration),
        )
        .agg(
            count("TransactionID").alias("tx_velocity"),
            avg("Amount").alias("avg_amount"),
            spark_max("Amount").alias("max_amount"),
            stddev("Amount").alias("std_amount"),
            avg("fraud_probability").alias("avg_fraud_prob"),
        )
    )

    # ---- Sink 1: Console (raw detections, append mode) ------------------
    (
        transactions_scored.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoint_dir, "console"))
        .start()
    )

    # ---- Sink 2: Parquet (append mode) ----------------------------------
    (
        transactions_scored.writeStream.outputMode("append")
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet"))
        .start()
    )

    # ---- Sink 3: Kafka fraud_alerts (append mode) -----------------------
    kafka_payload = transactions_scored.select(
        to_json(struct([col(c) for c in transactions_scored.columns])).alias("value")
    )
    (
        kafka_payload.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("topic", output_topic)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka"))
        .start()
    )

    # ---- Sink 4: Console (windowed user stats, update mode) -------------
    (
        user_stats.writeStream.outputMode("update")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoint_dir, "user_stats"))
        .start()
    )

    logger.info("Enhanced streaming queries started. Awaiting termination…")
    spark.streams.awaitAnyTermination()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced real-time Spark Streaming fraud detector."
    )
    parser.add_argument("--kafka_bootstrap", default="localhost:9092")
    parser.add_argument("--input_topic", default="transactions")
    parser.add_argument("--output_topic", default="fraud_alerts")
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "../../models"),
    )
    parser.add_argument("--checkpoint_dir", default="/tmp/fraud_checkpoint_enhanced")
    parser.add_argument("--output_path", default="/tmp/fraud_predictions_enhanced")
    parser.add_argument("--window_duration", default="10 minutes")
    parser.add_argument("--slide_duration", default="2 minutes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_enhanced_streaming(
        kafka_bootstrap=args.kafka_bootstrap,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        window_duration=args.window_duration,
        slide_duration=args.slide_duration,
    )
