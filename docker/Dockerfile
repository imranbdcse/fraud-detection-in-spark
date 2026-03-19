# =============================================================================
# Dockerfile
# Containerises the fraud detection application with Python, PySpark 3.5,
# Kafka client libraries, and all project dependencies.
# =============================================================================

FROM python:3.8-slim

# --------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# --------------------------------------------------------------------------
# Install Python dependencies
# --------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# Copy project source
# --------------------------------------------------------------------------
COPY . .

# Make shell scripts executable
RUN chmod +x scripts/start_services.sh

# --------------------------------------------------------------------------
# Environment defaults (overridable at runtime)
# --------------------------------------------------------------------------
ENV KAFKA_BROKER=kafka:9092 \
    ZOOKEEPER_QUORUM=zookeeper:2181 \
    KAFKA_TOPIC=transactions \
    MODEL_PATH=models/logistic_regression_fraud_model \
    FRAUD_OUTPUT_PATH=output/fraud_alerts \
    BATCH_INTERVAL=5 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

# --------------------------------------------------------------------------
# Default command: run the Kafka producer
# --------------------------------------------------------------------------
CMD ["python", "scripts/kafka_producer.py", "--broker", "kafka:9092", "--rate", "1"]
