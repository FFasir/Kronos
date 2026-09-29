#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${KRONOS_PROJECT_ROOT:-/root/kronos-strategy}"
PID_FILE="$PROJECT_ROOT/run/kronos-web.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Kronos web service is not running"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in {1..30}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "$PID_FILE"
echo "Kronos web service stopped"
