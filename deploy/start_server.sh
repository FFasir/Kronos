#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${KRONOS_PROJECT_ROOT:-/root/kronos-strategy}"
PID_FILE="$PROJECT_ROOT/run/kronos-web.pid"
LOG_FILE="$PROJECT_ROOT/logs/kronos-web.log"
ENV_FILE="$PROJECT_ROOT/.env.production"
PYTHON_BIN="${KRONOS_PYTHON:-/usr/local/miniconda3/envs/py312/bin/python}"
START_TIMEOUT_SECONDS="${KRONOS_START_TIMEOUT_SECONDS:-30}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing production environment file: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Kronos web service is already running (pid $(cat "$PID_FILE"))"
  exit 0
fi

mkdir -p "$PROJECT_ROOT/run" "$PROJECT_ROOT/logs"
rm -f "$PID_FILE"
cd "$PROJECT_ROOT"
nohup env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m uvicorn \
  src.service_layer.web_server:app --host 0.0.0.0 --port 8000 --workers 1 \
  >> "$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
pid="$(cat "$PID_FILE")"

for ((attempt = 1; attempt <= START_TIMEOUT_SECONDS; attempt++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Kronos web service exited during startup; see $LOG_FILE" >&2
    tail -n 40 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if "$PYTHON_BIN" -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=1)" >/dev/null 2>&1; then
    echo "Kronos web service started (pid $pid, health check passed)"
    exit 0
  fi
  sleep 1
done

echo "Kronos web service did not become healthy within ${START_TIMEOUT_SECONDS}s; see $LOG_FILE" >&2
exit 1
