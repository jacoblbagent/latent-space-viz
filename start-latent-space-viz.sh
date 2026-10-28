#!/usr/bin/env bash
# Serve the latent-space viewer on the first free port at or after 5205.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/web"
PORT=5205
while (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do
  echo "port $PORT busy, trying $((PORT + 1))"
  PORT=$((PORT + 1))
done
echo "http://127.0.0.1:${PORT}/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR"
