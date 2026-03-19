# Real-Time Credit Card Fraud Detection with Apache Spark

A complete machine learning pipeline and real-time streaming system for detecting credit card fraud using Apache Spark Structured Streaming and scikit-learn.

## Features

- **Synthetic data generation** – 100 k transactions with realistic 2% fraud ratio
- **Exploratory Data Analysis** – Jupyter notebooks with visualisations
- **Model training** – Logistic Regression with balanced class weights, ROC-AUC evaluation
- **Batch prediction** – Run inference on historical data files
- **Real-time streaming** – Spark Structured Streaming reads from Kafka and writes alerts
- **Enhanced streaming** – Windowed feature engineering (velocity, amount aggregations)
- **Helper scripts** – One-command start/stop for all components

## Project Structure

```
fraud-detection-in-spark/
├── data/
│   └── generate_synthetic_data.py   # Generates creditcard.csv
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── README.md
├── src/
│   ├── batch/
│   │   ├── train_model.py           # FraudDetectionTrainer class
│   │   ├── predict.py               # FraudPredictor class
│   │   └── batch_processor.py       # BatchProcessor class
│   └── streaming/
│       ├── fraud_detector.py        # Basic Spark streaming detector
│       ├── fraud_detector_enhanced.py  # Enhanced detector with feature engineering
│       └── README.md
├── models/                          # Saved model artefacts (auto-created)
├── tests/
│   └── test_end_to_end.py           # Integration test suite
├── scripts/
│   ├── start_streaming.sh
│   └── stop_streaming.sh
├── env/
│   ├── spark_config.conf
│   └── config.env
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.7+
- Apache Spark 2.4+ (with PySpark)
- Apache Kafka 2.8+
- Java 8+

## Installation

```bash
# Clone the repository
git clone https://github.com/imranbdcse/fraud-detection-in-spark.git
cd fraud-detection-in-spark

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate synthetic dataset

```bash
python data/generate_synthetic_data.py
```

### 2. Train the model

```bash
python src/batch/train_model.py \
    --data_path data/creditcard.csv \
    --model_dir models/ \
    --test_size 0.2
```

### 3. Run batch predictions

```bash
python src/batch/predict.py --model_dir models/ --input_csv data/creditcard.csv
```

### 4. Start real-time streaming (requires Kafka)

```bash
./scripts/start_streaming.sh --model-dir models/
```

Stop all components:

```bash
./scripts/stop_streaming.sh
```

### 5. Run integration tests

```bash
python -m pytest tests/test_end_to_end.py -v
```

## Notebooks

Launch Jupyter to explore the data and train the model interactively:

```bash
jupyter notebook notebooks/
```

See [notebooks/README.md](notebooks/README.md) for details.

## Streaming Architecture

```
Kafka Producer → Kafka "transactions" → Spark Streaming → Console / Parquet / Kafka "fraud_alerts"
```

See [src/streaming/README.md](src/streaming/README.md) for detailed instructions.

## License

MIT License – see [LICENSE](LICENSE).
