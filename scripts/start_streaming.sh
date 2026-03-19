#!/usr/bin/env bash
# start_streaming.sh
# Start all fraud detection streaming components in order.
# Usage: ./scripts/start_streaming.sh [--model-dir models/] [--checkpoint-dir /tmp/fraud_checkpoint]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/fraud_checkpoint}"
KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-localhost:9092}"
INPUT_TOPIC="${INPUT_TOPIC:-transactions}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-fraud_alerts}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-dir)     MODEL_DIR="$2";      shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --kafka-bootstrap) KAFKA_BOOTSTRAP="$2"; shift 2 ;;
    *) log "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---- 1. Check Kafka is reachable ----
log "Checking Kafka availability at ${KAFKA_BOOTSTRAP}…"
if ! command -v kafka-topics.sh &>/dev/null; then
  log "WARNING: kafka-topics.sh not found on PATH. Skipping Kafka health check."
else
  if ! kafka-topics.sh --list --bootstrap-server "${KAFKA_BOOTSTRAP}" &>/dev/null; then
    log "ERROR: Cannot connect to Kafka at ${KAFKA_BOOTSTRAP}. Is Kafka running?"
    exit 1
  fi
  log "Kafka is reachable."
fi

# ---- 2. Ensure topics exist ----
if command -v kafka-topics.sh &>/dev/null; then
  for topic in "${INPUT_TOPIC}" "${OUTPUT_TOPIC}"; do
    if ! kafka-topics.sh --describe --topic "${topic}" \
        --bootstrap-server "${KAFKA_BOOTSTRAP}" &>/dev/null; then
      log "Creating topic: ${topic}"
      kafka-topics.sh --create --topic "${topic}" \
          --bootstrap-server "${KAFKA_BOOTSTRAP}" \
          --partitions 1 --replication-factor 1
    else
      log "Topic already exists: ${topic}"
    fi
  done
fi

# ---- 3. Start Kafka producer in background ----
PRODUCER_SCRIPT="${PROJECT_ROOT}/scripts/kafka_producer.py"
if [[ -f "${PRODUCER_SCRIPT}" ]]; then
  log "Starting Kafka producer…"
  python3 "${PRODUCER_SCRIPT}" \
      --bootstrap_server "${KAFKA_BOOTSTRAP}" \
      --topic "${INPUT_TOPIC}" &
  PRODUCER_PID=$!
  log "Kafka producer started (PID: ${PRODUCER_PID})"
  echo "${PRODUCER_PID}" > /tmp/fraud_producer.pid
else
  log "WARNING: Kafka producer script not found at ${PRODUCER_SCRIPT}. Skipping."
fi

# ---- 4. Submit Spark streaming job ----
log "Submitting Spark streaming job…"
STREAMING_SCRIPT="${PROJECT_ROOT}/src/streaming/fraud_detector.py"

if ! command -v spark-submit &>/dev/null; then
  log "ERROR: spark-submit not found on PATH. Is Spark installed?"
  exit 1
fi

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \
  --properties-file "${PROJECT_ROOT}/env/spark_config.conf" \
  "${STREAMING_SCRIPT}" \
  --kafka_bootstrap "${KAFKA_BOOTSTRAP}" \
  --input_topic "${INPUT_TOPIC}" \
  --output_topic "${OUTPUT_TOPIC}" \
  --model_dir "${MODEL_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" &

SPARK_PID=$!
log "Spark streaming job started (PID: ${SPARK_PID})"
echo "${SPARK_PID}" > /tmp/fraud_spark.pid

log "All components started. PIDs:"
[[ -f /tmp/fraud_producer.pid ]] && log "  Kafka producer : $(cat /tmp/fraud_producer.pid)"
log "  Spark job      : ${SPARK_PID}"
log "To stop all components, run: ./scripts/stop_streaming.sh"
