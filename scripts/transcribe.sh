#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <audio-file>" >&2
  exit 2
fi

audio=$1
url="${AUDITOR_STT_URL:-http://localhost:8090}/inference"

curl --fail-with-body \
  -sS \
  -F "file=@${audio}" \
  -F "language=fi" \
  "$url"
