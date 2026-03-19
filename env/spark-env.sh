#!/usr/bin/env bash
# =============================================================================
# env/spark-env.sh
# Environment variables for Apache Spark 2.0 fraud detection project.
# Copy this file to $SPARK_HOME/conf/spark-env.sh or source it manually.
# =============================================================================

# Java home (adjust to your JDK installation)
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}"

# Spark installation directory
export SPARK_HOME="${SPARK_HOME:-/opt/spark}"

# Hadoop configuration (optional — needed for HDFS/YARN mode)
# export HADOOP_CONF_DIR=/etc/hadoop/conf

# Spark master URL (use 'yarn' for cluster mode)
export SPARK_MASTER_URL="${SPARK_MASTER_URL:-local[*]}"

# Driver and executor memory
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-2g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-2g}"

# Number of executor cores
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-2}"

# Log level (WARN reduces verbosity; use INFO for debugging)
export SPARK_LOG_LEVEL="${SPARK_LOG_LEVEL:-WARN}"

# Python executable used for PySpark
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-python3}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-python3}"

# Kafka broker address
export KAFKA_BROKER="${KAFKA_BROKER:-localhost:9092}"

# ZooKeeper quorum
export ZOOKEEPER_QUORUM="${ZOOKEEPER_QUORUM:-localhost:2181}"

# Kafka topic for transactions
export KAFKA_TOPIC="${KAFKA_TOPIC:-transactions}"

# Path to the trained fraud detection model
export MODEL_PATH="${MODEL_PATH:-models/logistic_regression_fraud_model}"

# Output path for fraud alerts
export FRAUD_OUTPUT_PATH="${FRAUD_OUTPUT_PATH:-output/fraud_alerts}"

# Spark Streaming batch interval (seconds)
export BATCH_INTERVAL="${BATCH_INTERVAL:-5}"

# Checkpoint directory for fault tolerance
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
