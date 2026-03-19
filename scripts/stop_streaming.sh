#!/usr/bin/env bash
# =============================================================================
# stop_streaming.sh
# Gracefully stop all Fraud Detection Streaming components.
#
# Usage:
#   chmod +x scripts/stop_streaming.sh
#   ./scripts/stop_streaming.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_DIR="${PROJECT_ROOT}/.pids"
LOG_FILE="${PROJECT_ROOT}/logs/streaming.log"

# ---- Helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

mkdir -p "${PROJECT_ROOT}/logs"

log "Stopping Fraud Detection Streaming components …"

# ---- Stop Spark Streaming job ----------------------------------------------
SPARK_PID_FILE="${PID_DIR}/spark.pid"
if [ -f "${SPARK_PID_FILE}" ]; then
  SPARK_PID=$(cat "${SPARK_PID_FILE}")
  if kill -0 "${SPARK_PID}" 2>/dev/null; then
    log "Sending SIGTERM to Spark job (PID=${SPARK_PID}) …"
    kill -TERM "${SPARK_PID}"
    # Wait up to 30 s for graceful shutdown
    for i in $(seq 1 30); do
      if ! kill -0 "${SPARK_PID}" 2>/dev/null; then
        log "Spark job stopped ✓"
        break
      fi
      sleep 1
    done
    if kill -0 "${SPARK_PID}" 2>/dev/null; then
      log "WARNING: Spark job did not stop gracefully; sending SIGKILL …"
      kill -9 "${SPARK_PID}" || true
    fi
  else
    log "Spark job (PID=${SPARK_PID}) is not running."
  fi
  rm -f "${SPARK_PID_FILE}"
else
  log "No Spark PID file found at ${SPARK_PID_FILE}."
fi

# ---- Stop Kafka producer ---------------------------------------------------
PRODUCER_PID_FILE="${PID_DIR}/producer.pid"
if [ -f "${PRODUCER_PID_FILE}" ]; then
  PRODUCER_PID=$(cat "${PRODUCER_PID_FILE}")
  if kill -0 "${PRODUCER_PID}" 2>/dev/null; then
    log "Stopping Kafka producer (PID=${PRODUCER_PID}) …"
    kill -TERM "${PRODUCER_PID}" || true
    sleep 2
    if kill -0 "${PRODUCER_PID}" 2>/dev/null; then
      kill -9 "${PRODUCER_PID}" || true
    fi
    log "Kafka producer stopped ✓"
  else
    log "Kafka producer (PID=${PRODUCER_PID}) is not running."
  fi
  rm -f "${PRODUCER_PID_FILE}"
else
  log "No producer PID file found at ${PRODUCER_PID_FILE}."
fi

# ---- Clean up checkpoint directories (optional) ----------------------------
# Uncomment below to remove checkpoint data on stop:
# rm -rf /tmp/fraud_detector_checkpoint /tmp/fraud_enhanced_checkpoint
# log "Checkpoint directories removed."

log "All components stopped."
