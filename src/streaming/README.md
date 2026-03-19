# Streaming Components

This directory contains the real-time Spark Structured Streaming scripts for
credit card fraud detection.

## Overview

```
src/streaming/
├── fraud_detector.py           Basic streaming fraud detector
└── fraud_detector_enhanced.py  Enhanced detector with feature engineering
```

### `fraud_detector.py`
- Reads JSON transaction messages from the Kafka topic `transactions`
- Applies the trained Logistic Regression model via a Pandas UDF
- Writes results to three sinks:
  - **Console** – live monitoring
  - **Parquet** – persistent storage
  - **Kafka** – `fraud_alerts` topic for downstream consumers

### `fraud_detector_enhanced.py`
- Includes all features of the basic detector
- Adds real-time feature engineering:
  - Transaction velocity (count per user in sliding window)
  - Amount aggregations (mean, max, std per user)
  - Hour-of-day from Unix timestamp
- Windowed aggregations with watermarking for fault tolerance
- Three risk levels: HIGH, MEDIUM, LOW
- Two output modes: append (detections) and update (user stats)

---

## Prerequisites

1. Apache Kafka running on `localhost:9092`
2. Kafka topic `transactions` created
3. Trained model artefacts in `models/` (run `src/batch/train_model.py` first)
4. Apache Spark with the Kafka connector package

## Running the Basic Detector

```bash
# 1. Source environment variables
source env/config.env

# 2. Submit the Spark job
spark-submit \
    --name FraudDetector \
    --properties-file env/spark_config.conf \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    src/streaming/fraud_detector.py
```

## Running the Enhanced Detector

```bash
spark-submit \
    --name FraudDetectorEnhanced \
    --properties-file env/spark_config.conf \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    src/streaming/fraud_detector_enhanced.py
```

## Using the Helper Scripts

```bash
# Start all components (Kafka producer + Spark streaming)
chmod +x scripts/start_streaming.sh scripts/stop_streaming.sh
./scripts/start_streaming.sh

# Start with the enhanced detector
./scripts/start_streaming.sh --enhanced

# Start without the Kafka producer (if you manage it separately)
./scripts/start_streaming.sh --no-producer

# Stop all components
./scripts/stop_streaming.sh
```

## Overriding Configuration

All parameters can be changed via environment variables:

| Variable                  | Default                    | Description                          |
|---------------------------|----------------------------|--------------------------------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092`           | Kafka broker address                 |
| `KAFKA_INPUT_TOPIC`       | `transactions`             | Topic to read from                   |
| `KAFKA_OUTPUT_TOPIC`      | `fraud_alerts`             | Topic to write alerts to             |
| `MODEL_PATH`              | `models/fraud_model.pkl`   | Path to trained model                |
| `SCALER_PATH`             | `models/scaler.pkl`        | Path to fitted scaler                |
| `CHECKPOINT_DIR`          | `/tmp/fraud_detector_checkpoint` | Spark checkpoint location      |
| `OUTPUT_DIR`              | `/tmp/fraud_detector_output`     | Parquet output directory       |
| `WINDOW_DURATION`         | `10 minutes`               | Aggregation window (enhanced only)   |
| `SLIDE_DURATION`          | `5 minutes`                | Window slide interval (enhanced)     |
| `WATERMARK_DELAY`         | `5 minutes`                | Late-data tolerance (enhanced)       |

Example:

```bash
export KAFKA_BOOTSTRAP_SERVERS=kafka-broker:9092
export WINDOW_DURATION="5 minutes"
spark-submit ... src/streaming/fraud_detector_enhanced.py
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: Model not found` | Model not trained | Run `python src/batch/train_model.py` |
| `Connection refused` to Kafka | Kafka not running | Start Zookeeper + Kafka broker |
| `ClassNotFoundException` for Kafka connector | Missing `--packages` flag | Add `--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0` |
| Checkpoint directory errors | Stale checkpoint state | Delete `/tmp/fraud_*_checkpoint` and restart |
| Out-of-memory errors | Large micro-batches | Reduce `spark.executor.memory` or increase `trigger` interval |

## Performance Tuning Tips

1. **Increase executor memory** for large message volumes:
   ```
   spark.executor.memory=4g
   ```

2. **Adjust shuffle partitions** to match your cluster:
   ```
   spark.sql.shuffle.partitions=16
   ```

3. **Tune trigger interval** for latency vs. throughput trade-off:
   - Lower (`10 seconds`) → lower latency
   - Higher (`60 seconds`) → higher throughput

4. **Use Kafka partitions** for parallelism — more partitions allow more
   Spark tasks to run in parallel.

5. **Enable Kryo serialisation** (already set in `spark_config.conf`) for
   faster data shuffles.
