"""
Enhanced Real-time Spark Structured Streaming fraud detector.

Extends the basic ``fraud_detector.py`` with real-time feature engineering:

  * Transaction velocity  (transactions per user in a sliding window)
  * Amount aggregations   (mean, max, std per user window)
  * Time-based features   (hour of day from Unix timestamp)

Supports two output modes:
  * ``append``  — new fraud detections
  * ``update``  — aggregated per-user statistics

Optimisations:
  * Model and scaler are broadcast to all executors.
  * Watermarking prevents unbounded state accumulation.
  * Explicit partitioning improves throughput.

Usage
-----
    spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \\
        src/streaming/fraud_detector_enhanced.py

Environment variables
---------------------
    KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
    KAFKA_INPUT_TOPIC         default: transactions
    KAFKA_OUTPUT_TOPIC        default: fraud_alerts
    MODEL_PATH                default: models/fraud_model.pkl
    SCALER_PATH               default: models/scaler.pkl
    CHECKPOINT_DIR            default: /tmp/fraud_enhanced_checkpoint
    OUTPUT_DIR                default: /tmp/fraud_enhanced_output
    WINDOW_DURATION           default: 10 minutes
    SLIDE_DURATION            default: 5 minutes
    WATERMARK_DELAY           default: 5 minutes
"""

import logging
import os
import sys

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
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/tmp/fraud_enhanced_checkpoint")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/fraud_enhanced_output")
WINDOW_DURATION = os.getenv("WINDOW_DURATION", "10 minutes")
SLIDE_DURATION = os.getenv("SLIDE_DURATION", "5 minutes")
WATERMARK_DELAY = os.getenv("WATERMARK_DELAY", "5 minutes")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load and return the trained model and scaler."""
    import joblib

    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefact not found: {path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model loaded: %s", model_path)
    logger.info("Scaler loaded: %s", scaler_path)
    return model, scaler


def build_schema():
    """Build the StructType for incoming transaction JSON."""
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    fields = [
        StructField("TransactionID", StringType(), True),
        StructField("UserID", StringType(), True),
    ]
    for col_name in FEATURE_COLUMNS:
        fields.append(StructField(col_name, DoubleType(), True))
    fields.append(StructField("Timestamp", DoubleType(), True))
    return StructType(fields)


def build_spark_session(app_name: str = "FraudDetectorEnhanced"):
    """Create a SparkSession with optimised settings."""
    from pyspark.sql import SparkSession

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


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def add_time_features(df):
    """Extract hour-of-day from the Unix ``Timestamp`` column."""
    from pyspark.sql.functions import col, from_unixtime, hour

    return df.withColumn(
        "event_time", from_unixtime(col("Timestamp").cast("long"))
    ).withColumn("hour_of_day", hour("event_time"))


def add_windowed_features(df):
    """
    Compute per-user windowed aggregations:

    * ``tx_count``    — transaction count in the window
    * ``amount_mean`` — mean Amount per user
    * ``amount_max``  — max Amount per user
    * ``amount_std``  — std dev of Amount per user
    """
    from pyspark.sql.functions import avg, col, count, max as spark_max, stddev, window

    agg_df = (
        df.withWatermark("event_time", WATERMARK_DELAY)
        .groupBy(
            col("UserID"),
            window(col("event_time"), WINDOW_DURATION, SLIDE_DURATION),
        )
        .agg(
            count("TransactionID").alias("tx_count"),
            avg("Amount").alias("amount_mean"),
            spark_max("Amount").alias("amount_max"),
            stddev("Amount").alias("amount_std"),
        )
        .withColumn("window_start", col("window.start"))
        .withColumn("window_end", col("window.end"))
        .drop("window")
    )
    return agg_df


# --------------------------------------------------------------------------- #
# Model application
# --------------------------------------------------------------------------- #

def apply_fraud_model(df, model, scaler):
    """Apply the fraud model via a Pandas UDF (vectorised)."""
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    sc = df.sparkSession.sparkContext
    bc_model = sc.broadcast(model)
    bc_scaler = sc.broadcast(scaler)

    @pandas_udf(DoubleType())
    def predict_probability(*cols):
        X = pd.concat(list(cols), axis=1)
        X.columns = FEATURE_COLUMNS
        X_scaled = bc_scaler.value.transform(X.fillna(0).values)
        probs = bc_model.value.predict_proba(X_scaled)[:, 1]
        return pd.Series(probs)

    feature_cols = [df[c] for c in FEATURE_COLUMNS]
    return df.withColumn("fraud_probability", predict_probability(*feature_cols))


def add_risk_label(df):
    """Assign a string risk label based on fraud_probability."""
    from pyspark.sql.functions import col, when

    return df.withColumn(
        "risk_label",
        when(col("fraud_probability") >= 0.8, "HIGH")
        .when(col("fraud_probability") >= 0.5, "MEDIUM")
        .otherwise("LOW"),
    ).withColumn(
        "is_fraud",
        when(col("fraud_probability") >= 0.5, 1).otherwise(0),
    )


# --------------------------------------------------------------------------- #
# Output sinks
# --------------------------------------------------------------------------- #

def write_fraud_detections(df, checkpoint_dir: str, output_dir: str):
    """Write new fraud detections (append mode) to Parquet and console."""
    from pyspark.sql.functions import col

    fraud_df = df.filter(col("is_fraud") == 1)

    parquet_query = (
        fraud_df.writeStream.outputMode("append")
        .format("parquet")
        .option("path", os.path.join(output_dir, "fraud_detections"))
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet_detections"))
        .trigger(processingTime="30 seconds")
        .start()
    )

    console_query = (
        fraud_df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoint_dir, "console"))
        .start()
    )

    return [parquet_query, console_query]


def write_aggregated_stats(agg_df, checkpoint_dir: str, output_dir: str):
    """Write per-user windowed aggregations (update mode) to Parquet."""
    return (
        agg_df.writeStream.outputMode("update")
        .format("parquet")
        .option("path", os.path.join(output_dir, "user_stats"))
        .option("checkpointLocation", os.path.join(checkpoint_dir, "parquet_stats"))
        .trigger(processingTime="60 seconds")
        .start()
    )


def write_alerts_to_kafka(df, checkpoint_dir: str):
    """Write high-risk alerts back to Kafka as JSON."""
    from pyspark.sql.functions import col, struct, to_json

    alert_df = (
        df.filter(col("risk_label") == "HIGH")
        .select(
            to_json(
                struct(
                    col("TransactionID"),
                    col("UserID"),
                    col("fraud_probability"),
                    col("risk_label"),
                    col("event_time"),
                )
            ).alias("value")
        )
    )

    return (
        alert_df.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", KAFKA_OUTPUT_TOPIC)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka"))
        .start()
    )


# --------------------------------------------------------------------------- #
# Entry-point
# --------------------------------------------------------------------------- #

def main():
    logger.info("Starting Enhanced Fraud Detector …")
    logger.info("  Kafka bootstrap : %s", KAFKA_BOOTSTRAP_SERVERS)
    logger.info("  Input topic     : %s", KAFKA_INPUT_TOPIC)
    logger.info("  Output topic    : %s", KAFKA_OUTPUT_TOPIC)
    logger.info("  Window          : %s / slide %s", WINDOW_DURATION, SLIDE_DURATION)

    try:
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    except FileNotFoundError as exc:
        logger.error("%s — run src/batch/train_model.py first", exc)
        sys.exit(1)

    from pyspark.sql.functions import col, from_json

    spark = build_spark_session()
    schema = build_schema()

    # Read from Kafka
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON
    transactions = (
        raw_stream.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_ts"),
        )
        .select("data.*", "kafka_ts")
    )

    # Feature engineering
    transactions = add_time_features(transactions)

    # Score transactions
    scored = apply_fraud_model(transactions, model, scaler)
    final = add_risk_label(scored)

    # Windowed aggregations
    agg_stats = add_windowed_features(transactions)

    # Write outputs
    queries = []
    queries.extend(write_fraud_detections(final, CHECKPOINT_DIR, OUTPUT_DIR))
    queries.append(write_aggregated_stats(agg_stats, CHECKPOINT_DIR, OUTPUT_DIR))
    queries.append(write_alerts_to_kafka(final, CHECKPOINT_DIR))

    logger.info("%d streaming queries started. Awaiting termination …", len(queries))
    for q in queries:
        q.awaitTermination()


if __name__ == "__main__":
    main()
