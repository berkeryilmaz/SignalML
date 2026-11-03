#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/backend"

export PYTHONPATH="${BACKEND_DIR}"
cd "${BACKEND_DIR}"

# Allow port to be specified as argument or PORT env var, defaulting to 8080
TARGET_PORT="${1:-${PORT:-8080}}"

# Check if target port is occupied; if 8080 is in use, fallback to 8081
if python3 -c "import socket; s = socket.socket(); err = s.connect_ex(('127.0.0.1', int('$TARGET_PORT'))); s.close(); exit(0 if err == 0 else 1)" 2>/dev/null; then
    if [ "$TARGET_PORT" = "8080" ]; then
        echo "Port 8080 is currently occupied by another service. Switching to port 8081..."
        TARGET_PORT=8081
    fi
fi

echo "Starting SignalML on http://0.0.0.0:${TARGET_PORT}..."
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${TARGET_PORT}"
