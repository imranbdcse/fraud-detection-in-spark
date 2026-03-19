# fraud-detection-in-spark

> **Real-time credit card fraud detection using Apache Spark 3.5, Kafka, and PySpark MLlib**

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PySpark 3.5.8](https://img.shields.io/badge/pyspark-3.5.8-orange.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Usage](#usage)
   - [1. Train the Model](#1-train-the-model)
   - [2. Start the Kafka Producer](#2-start-the-kafka-producer)
   - [3. Run the Spark Streaming Job](#3-run-the-spark-streaming-job)
   - [4. Run Batch Processing](#4-run-batch-processing)
   - [5. Docker Compose (All-in-One)](#5-docker-compose-all-in-one)
7. [Testing](#testing)
8. [Configuration](#configuration)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview

This project demonstrates a **production-grade, real-time fraud detection pipeline** for credit card transactions. It uses:

- **Apache Kafka** to ingest a continuous stream of synthetic transaction events.
- **Apache Spark Structured Streaming** (`readStream` / `writeStream.foreachBatch`) to consume, parse, and process the stream in micro-batches.
- **Spark MLlib** (Logistic Regression with cross-validation) to classify each transaction as fraudulent or legitimate.
- **Jupyter Notebooks** for interactive EDA and model training on the Kaggle Credit Card Fraud dataset.
- **Docker Compose** to spin up all required services with a single command.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                           │
│   scripts/kafka_producer.py  →  Kafka topic: "transactions"      │
└──────────────────────────────┬───────────────────────────────────┘
                               │  Kafka Stream
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Spark Streaming Layer                          │
│   src/streaming/stream_processor.py                              │
│   • Parse JSON  →  Encode features  →  Assemble vectors          │
│   • Load pretrained LogisticRegressionModel                       │
│   • Predict: fraud (1) / legitimate (0)                           │
└──────────────────────────────┬───────────────────────────────────┘
                               │
              ┌────────────────┴─────────────────┐
              ▼                                   ▼
   output/fraud_alerts/             (extend: DB / alert API)
   (JSON files per micro-batch)
```

---

## Project Structure

```
fraud-detection-in-spark/
├── data/                        # Dataset directory (add creditcard.csv here)
├── notebooks/
│   └── model_training.ipynb     # EDA + Logistic Regression training notebook
├── src/
│   ├── streaming/
│   │   └── stream_processor.py  # Spark Streaming consumer + fraud inference
│   └── batch/
│       └── batch_processor.py   # Offline batch scoring with Spark
├── models/                      # Saved Spark ML models
├── tests/
│   ├── test_kafka_producer.py   # Unit tests for the Kafka producer
│   └── test_stream_processor.py # Unit tests for streaming helpers
├── scripts/
│   ├── kafka_producer.py        # Transaction stream simulator
│   └── start_services.sh        # Helper script to start all services
├── env/
│   └── spark-env.sh             # Environment variable configuration
├── docker/
│   ├── Dockerfile               # Container image definition
│   └── docker-compose.yml       # Multi-service orchestration
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.8+ |
| Apache Spark | 3.5.x |
| Apache Kafka | 2.x |
| Java (JDK) | 8, 11 or 17 |
| Docker & Docker Compose | Latest |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/imranbdcse/fraud-detection-in-spark.git
cd fraud-detection-in-spark
```

### 2. Create a Python virtual environment

```bash
python3 -m venv env
source env/bin/activate          # Linux / macOS
# env\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp env/spark-env.sh /opt/spark/conf/spark-env.sh   # or source it manually
source env/spark-env.sh
```

### 5. Download the dataset *(for model training only)*

Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory.

---

## Usage

### 1. Train the Model

Open the Jupyter notebook to perform EDA and train the Logistic Regression model:

```bash
jupyter notebook notebooks/model_training.ipynb
```

Or train via `spark-submit`:

```bash
spark-submit src/batch/batch_processor.py \
    --input-path data/creditcard.csv \
    --model-path models/logistic_regression_fraud_model \
    --mode train
```

### 2. Start the Kafka Producer

Simulate a live stream of credit card transactions (2 msg/s, 5% fraud rate):

```bash
python scripts/kafka_producer.py \
    --topic transactions \
    --broker localhost:9092 \
    --rate 2 \
    --fraud-ratio 0.05
```

### 3. Run the Spark Streaming Job

```bash
spark-submit \
    --master local[*] \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \
    src/streaming/stream_processor.py \
    --topic transactions \
    --broker localhost:9092 \
    --model-path models/logistic_regression_fraud_model \
    --output-path output/fraud_alerts
```

Fraudulent transactions are written as JSON files to `output/fraud_alerts/`.

### 4. Run Batch Processing

Score a historical dataset with the pretrained model:

```bash
spark-submit src/batch/batch_processor.py \
    --input-path data/creditcard.csv \
    --model-path models/logistic_regression_fraud_model \
    --output-path output/batch_predictions \
    --mode predict
```

### 5. Docker Compose (All-in-One)

> **Note:** Train and save the model first (`models/logistic_regression_fraud_model` must exist).

```bash
cd docker
docker-compose up --build
```

This starts ZooKeeper, Kafka, the transaction producer, and the Spark Streaming job automatically.

---

## Testing

Run the unit test suite from the project root:

```bash
pytest tests/ -v
```

Tests cover:
- `test_kafka_producer.py` — transaction generation, JSON serialisation, argument parsing, producer send logic.
- `test_stream_processor.py` — JSON parsing, feature encoding, mapping correctness.

---

## Configuration

All environment variables can be set in `env/spark-env.sh` or overridden at runtime:

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BROKER` | `localhost:9092` | Kafka broker address |
| `ZOOKEEPER_QUORUM` | `localhost:2181` | ZooKeeper quorum |
| `KAFKA_TOPIC` | `transactions` | Topic name |
| `MODEL_PATH` | `models/logistic_regression_fraud_model` | Trained model path |
| `FRAUD_OUTPUT_PATH` | `output/fraud_alerts` | Fraud alert output directory |
| `BATCH_INTERVAL` | `5` | Streaming micro-batch size (seconds) |
| `SPARK_DRIVER_MEMORY` | `2g` | Spark driver memory |
| `SPARK_EXECUTOR_MEMORY` | `2g` | Spark executor memory |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a pull request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
