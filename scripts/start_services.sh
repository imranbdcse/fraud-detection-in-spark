#!/usr/bin/env bash
# =============================================================================
# start_services.sh
# Script to start all required services for the fraud detection project.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

KAFKA_HOME="${KAFKA_HOME:-/opt/kafka}"
SPARK_HOME="${SPARK_HOME:-/opt/spark}"
ZOOKEEPER_PORT=2181
KAFKA_PORT=9092
KAFKA_TOPIC="transactions"

# Color codes for output
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

log_info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# 1. Start ZooKeeper
# ---------------------------------------------------------------------------
start_zookeeper() {
    log_info "Starting ZooKeeper..."
    if lsof -i :"$ZOOKEEPER_PORT" &>/dev/null; then
        log_warn "ZooKeeper already running on port $ZOOKEEPER_PORT."
    else
        "$KAFKA_HOME/bin/zookeeper-server-start.sh" \
            "$KAFKA_HOME/config/zookeeper.properties" &>/dev/null &
        sleep 3
        log_info "ZooKeeper started."
    fi
}

# ---------------------------------------------------------------------------
# 2. Start Kafka Broker
# ---------------------------------------------------------------------------
start_kafka() {
    log_info "Starting Kafka broker..."
    if lsof -i :"$KAFKA_PORT" &>/dev/null; then
        log_warn "Kafka already running on port $KAFKA_PORT."
    else
        "$KAFKA_HOME/bin/kafka-server-start.sh" \
            "$KAFKA_HOME/config/server.properties" &>/dev/null &
        sleep 5
        log_info "Kafka broker started."
    fi
}

# ---------------------------------------------------------------------------
# 3. Create Kafka topic (idempotent)
# ---------------------------------------------------------------------------
create_kafka_topic() {
    log_info "Creating Kafka topic '$KAFKA_TOPIC' (if not exists)..."
    "$KAFKA_HOME/bin/kafka-topics.sh" \
        --create \
        --if-not-exists \
        --topic "$KAFKA_TOPIC" \
        --partitions 3 \
        --replication-factor 1 \
        --bootstrap-server "localhost:$KAFKA_PORT"
    log_info "Kafka topic '$KAFKA_TOPIC' ready."
}

# ---------------------------------------------------------------------------
# 4. Start Kafka producer (background)
# ---------------------------------------------------------------------------
start_producer() {
    log_info "Starting Kafka transaction producer..."
    python "$PROJECT_ROOT/scripts/kafka_producer.py" \
        --topic "$KAFKA_TOPIC" \
        --broker "localhost:$KAFKA_PORT" \
        --rate 2 \
        --fraud-ratio 0.05 &
    PRODUCER_PID=$!
    log_info "Producer started (PID: $PRODUCER_PID)."
}

# ---------------------------------------------------------------------------
# 5. Submit Spark Streaming job
# ---------------------------------------------------------------------------
start_spark_streaming() {
    log_info "Submitting Spark Streaming job..."
    "$SPARK_HOME/bin/spark-submit" \
        --master local[*] \
        --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 \
        "$PROJECT_ROOT/src/streaming/stream_processor.py" \
        --topic "$KAFKA_TOPIC" \
        --broker "localhost:$KAFKA_PORT" \
        --model-path "$PROJECT_ROOT/models/logistic_regression_fraud_model" &
    log_info "Spark Streaming job submitted."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log_info "=== Fraud Detection Services Startup ==="
    start_zookeeper
    start_kafka
    create_kafka_topic
    start_producer
    start_spark_streaming
    log_info "All services started. Press Ctrl+C to stop."
    wait
}

main "$@"
