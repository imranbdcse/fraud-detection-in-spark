#!/usr/bin/env bash
# =============================================================================
# start_streaming.sh
# Start all components of the Fraud Detection Streaming Pipeline.
#
# Order:
#   1. Verify Kafka is reachable
#   2. Start the Kafka transaction producer (background)
#   3. Submit the Spark Structured Streaming job
#
# Usage:
#   chmod +x scripts/start_streaming.sh
#   ./scripts/start_streaming.sh
#   ./scripts/start_streaming.sh --enhanced     # use enhanced detector
#   ./scripts/start_streaming.sh --no-producer  # skip producer start
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables
# shellcheck source=../env/config.env
source "${PROJECT_ROOT}/env/config.env"

# ---- Defaults ---------------------------------------------------------------
ENHANCED=false
START_PRODUCER=true
LOG_FILE="${PROJECT_ROOT}/logs/streaming.log"
PID_DIR="${PROJECT_ROOT}/.pids"

# ---- Argument parsing -------------------------------------------------------
for arg in "$@"; do
  case "${arg}" in
    --enhanced)    ENHANCED=true ;;
    --no-producer) START_PRODUCER=false ;;
    --help|-h)
      echo "Usage: $0 [--enhanced] [--no-producer]"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

# ---- Helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
fail() { log "ERROR: $*"; exit 1; }

mkdir -p "${PROJECT_ROOT}/logs" "${PID_DIR}"

# ---- 1. Check Kafka ---------------------------------------------------------
log "Checking Kafka connectivity at ${KAFKA_BOOTSTRAP_SERVERS} …"

KAFKA_HOST="${KAFKA_BOOTSTRAP_SERVERS%%:*}"
KAFKA_PORT="${KAFKA_BOOTSTRAP_SERVERS##*:}"

if ! nc -z -w 5 "${KAFKA_HOST}" "${KAFKA_PORT}" 2>/dev/null; then
  fail "Cannot reach Kafka at ${KAFKA_BOOTSTRAP_SERVERS}. Is the broker running?"
fi
log "Kafka is reachable ✓"

# ---- 2. Start Kafka producer (optional) ------------------------------------
if [ "${START_PRODUCER}" = true ]; then
  PRODUCER_SCRIPT="${PROJECT_ROOT}/scripts/kafka_producer.py"

  if [ ! -f "${PRODUCER_SCRIPT}" ]; then
    log "WARNING: Kafka producer not found at ${PRODUCER_SCRIPT}; skipping."
  else
    log "Starting Kafka producer …"
    nohup python3 "${PRODUCER_SCRIPT}" \
      >> "${PROJECT_ROOT}/logs/producer.log" 2>&1 &
    PRODUCER_PID=$!
    echo "${PRODUCER_PID}" > "${PID_DIR}/producer.pid"
    log "Kafka producer started (PID=${PRODUCER_PID})"
    sleep 2  # give the producer a moment to connect
  fi
fi

# ---- 3. Submit Spark Streaming job -----------------------------------------
if [ "${ENHANCED}" = true ]; then
  DETECTOR_SCRIPT="${PROJECT_ROOT}/src/streaming/fraud_detector_enhanced.py"
  APP_NAME="FraudDetectorEnhanced"
else
  DETECTOR_SCRIPT="${PROJECT_ROOT}/src/streaming/fraud_detector.py"
  APP_NAME="FraudDetector"
fi

if [ ! -f "${DETECTOR_SCRIPT}" ]; then
  fail "Streaming script not found: ${DETECTOR_SCRIPT}"
fi

SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"
if [ ! -x "${SPARK_SUBMIT}" ]; then
  # Try spark-submit on PATH
  SPARK_SUBMIT="spark-submit"
fi

KAFKA_PACKAGE="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

log "Submitting Spark job: ${APP_NAME}"
log "  Script  : ${DETECTOR_SCRIPT}"
log "  Package : ${KAFKA_PACKAGE}"

nohup "${SPARK_SUBMIT}" \
  --name "${APP_NAME}" \
  --properties-file "${PROJECT_ROOT}/env/spark_config.conf" \
  --packages "${KAFKA_PACKAGE}" \
  "${DETECTOR_SCRIPT}" \
  >> "${PROJECT_ROOT}/logs/spark_streaming.log" 2>&1 &

SPARK_PID=$!
echo "${SPARK_PID}" > "${PID_DIR}/spark.pid"
log "Spark streaming job started (PID=${SPARK_PID})"

log "All components started. Logs:"
log "  Producer : ${PROJECT_ROOT}/logs/producer.log"
log "  Spark    : ${PROJECT_ROOT}/logs/spark_streaming.log"
log ""
log "To stop: ./scripts/stop_streaming.sh"
