#!/usr/bin/env bash
# stop_streaming.sh
# Gracefully stop all fraud detection streaming components.
# Usage: ./scripts/stop_streaming.sh

set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

stop_pid() {
  local pid_file="$1"
  local label="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      log "Stopping ${label} (PID: ${pid})…"
      kill -SIGTERM "${pid}" || true
      # Wait up to 10 s for graceful shutdown
      for _ in {1..10}; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      # Force-kill if still running
      if kill -0 "${pid}" 2>/dev/null; then
        log "Force-killing ${label} (PID: ${pid})…"
        kill -9 "${pid}" || true
      fi
      log "${label} stopped."
    else
      log "${label} (PID: ${pid}) is not running."
    fi
    rm -f "${pid_file}"
  else
    log "No PID file found for ${label} (${pid_file}). Skipping."
  fi
}

log "Stopping all fraud detection streaming components…"

stop_pid /tmp/fraud_spark.pid    "Spark streaming job"
stop_pid /tmp/fraud_producer.pid "Kafka producer"

# ---- Clean up checkpoint temp files (optional) ----
if [[ -d /tmp/fraud_checkpoint ]]; then
  log "Removing checkpoint directory /tmp/fraud_checkpoint…"
  rm -rf /tmp/fraud_checkpoint
fi

log "All components stopped. Resources cleaned up."
