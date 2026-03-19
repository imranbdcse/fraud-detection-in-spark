# Streaming Components

This directory contains Spark Structured Streaming scripts for real-time credit card fraud detection.

## Overview

The streaming architecture follows this data flow:

```
Kafka Producer
     │
     ▼
Kafka Topic: "transactions"
     │
     ▼
Spark Structured Streaming (fraud_detector.py)
     │
     ├──▶ Console output (monitoring)
     ├──▶ Parquet files  (storage)
     └──▶ Kafka Topic: "fraud_alerts" (downstream systems)
```

## Scripts

| Script | Description |
|---|---|
| `fraud_detector.py` | Basic streaming detector: reads from Kafka, applies ML model, writes to console / Parquet / Kafka. |
| `fraud_detector_enhanced.py` | Enhanced detector with real-time feature engineering (transaction velocity, amount aggregations) and windowed statistics. |

## Prerequisites

1. Apache Spark 2.4+ with PySpark
2. Apache Kafka with topics `transactions` and `fraud_alerts`
3. Trained model files in `models/` (`fraud_model.pkl`, `scaler.pkl`)

## Running the Basic Detector

```bash
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.8 \
    --properties-file env/spark_config.conf \
    src/streaming/fraud_detector.py \
    --kafka_bootstrap localhost:9092 \
    --input_topic transactions \
    --output_topic fraud_alerts \
    --model_dir models/ \
    --checkpoint_dir /tmp/fraud_checkpoint \
    --output_path /tmp/fraud_predictions
```

## Running the Enhanced Detector

```bash
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.8 \
    --properties-file env/spark_config.conf \
    src/streaming/fraud_detector_enhanced.py \
    --kafka_bootstrap localhost:9092 \
    --input_topic transactions \
    --output_topic fraud_alerts \
    --model_dir models/ \
    --checkpoint_dir /tmp/fraud_checkpoint_enhanced \
    --output_path /tmp/fraud_predictions_enhanced \
    --window_duration "10 minutes" \
    --slide_duration "2 minutes"
```

## Helper Scripts

Start all components at once:

```bash
./scripts/start_streaming.sh --model-dir models/ --checkpoint-dir /tmp/fraud_checkpoint
```

Stop all components gracefully:

```bash
./scripts/stop_streaming.sh
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `ClassNotFoundException: kafka...` | Ensure `--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.8` is passed to `spark-submit`. |
| `Connection refused` on Kafka | Verify Kafka is running: `kafka-topics.sh --list --bootstrap-server localhost:9092` |
| Model not found | Run `python src/batch/train_model.py` to train and save the model first. |
| Out of memory errors | Increase `spark.driver.memory` and `spark.executor.memory` in `env/spark_config.conf`. |
| Checkpoint corruption | Delete the checkpoint directory and restart: `rm -rf /tmp/fraud_checkpoint` |

## Performance Tuning

- **Increase parallelism**: Set `spark.sql.shuffle.partitions` to 2–4× the number of executor cores.
- **Broadcast model**: Both detectors broadcast the model and scaler to avoid repeated serialization.
- **Tune watermark**: Adjust the watermark delay in `fraud_detector_enhanced.py` to balance latency vs. completeness.
- **Kafka partitions**: Increase Kafka topic partitions to match executor count for higher throughput.
