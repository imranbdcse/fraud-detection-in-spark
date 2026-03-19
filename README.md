# Real-Time Credit Card Fraud Detection with Apache Spark

A production-ready pipeline for detecting credit card fraud in real time using
Apache Spark Structured Streaming, Apache Kafka, and Scikit-learn.

## Project Structure

```
fraud-detection-in-spark/
├── data/
│   └── generate_synthetic_data.py   # Synthetic dataset generator
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── README.md
├── src/
│   ├── batch/
│   │   ├── train_model.py           # Training pipeline
│   │   ├── predict.py               # Inference helper
│   │   └── batch_processor.py       # Batch processing
│   └── streaming/
│       ├── fraud_detector.py        # Basic Spark streaming job
│       ├── fraud_detector_enhanced.py # Enhanced with feature engineering
│       └── README.md
├── models/                          # Saved model artefacts (generated)
├── tests/
│   └── test_end_to_end.py           # Integration tests
├── scripts/
│   ├── start_streaming.sh
│   └── stop_streaming.sh
├── env/
│   ├── config.env                   # Environment variables
│   └── spark_config.conf            # Spark configuration
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.7+
- Apache Spark 2.4+ (with Kafka connector)
- Apache Kafka 2.8+
- Java 8+

## Installation

```bash
# Clone the repository
git clone https://github.com/imranbdcse/fraud-detection-in-spark.git
cd fraud-detection-in-spark

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Data

```bash
python data/generate_synthetic_data.py
```

Creates `data/creditcard.csv` with 100 000 transactions (2 % fraud rate).

### 2. Train the Model

```bash
python src/batch/train_model.py
# With custom paths:
python src/batch/train_model.py \
    --data-path data/creditcard.csv \
    --model-dir models/ \
    --test-size 0.2
```

Saves `models/fraud_model.pkl` and `models/scaler.pkl`.

### 3. Run Batch Predictions

```bash
python src/batch/predict.py --input data/creditcard.csv --output predictions.csv
```

### 4. Run Batch Processing

```bash
python src/batch/batch_processor.py \
    --input data/creditcard.csv \
    --format csv \
    --output-dir batch_results/
```

### 5. Start Real-Time Streaming

```bash
# Source environment variables
source env/config.env

# Start all components (Kafka producer + Spark streaming)
chmod +x scripts/start_streaming.sh scripts/stop_streaming.sh
./scripts/start_streaming.sh

# Or submit Spark job directly
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    --properties-file env/spark_config.conf \
    src/streaming/fraud_detector.py

# Stop all components
./scripts/stop_streaming.sh
```

### 6. Run Notebooks

```bash
jupyter notebook notebooks/
```

Open `01_exploratory_data_analysis.ipynb` then `02_model_training.ipynb`.

## Running Tests

```bash
pytest tests/test_end_to_end.py -v
```

## Configuration

| File | Purpose |
|------|---------|
| `env/config.env` | Kafka, model, and path environment variables |
| `env/spark_config.conf` | Spark memory, executor, and streaming settings |

Key environment variables (see `env/config.env` for full list):

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_INPUT_TOPIC=transactions
KAFKA_OUTPUT_TOPIC=fraud_alerts
MODEL_PATH=models/fraud_model.pkl
SCALER_PATH=models/scaler.pkl
```

## Data Flow

```
Kafka Producer
      │
      ▼
Kafka topic: transactions
      │
      ▼
Spark Structured Streaming
  ├── Parse JSON
  ├── Feature engineering (enhanced)
  ├── Apply ML model (Pandas UDF)
  └── Write to sinks:
        ├── Console (monitoring)
        ├── Parquet (storage)
        └── Kafka topic: fraud_alerts
```

## Model Details

- **Algorithm**: Logistic Regression with `class_weight='balanced'`
- **Scaler**: StandardScaler
- **Evaluation**: ROC-AUC, F1-Score, Precision, Recall, cross-validation
- **Serialisation**: joblib

## License

MIT License — see [LICENSE](LICENSE) for details.
