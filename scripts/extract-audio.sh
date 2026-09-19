#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <video-file> <audio-file>" >&2
  exit 2
fi

input=$1
output=$2

ffmpeg -hide_banner -loglevel error \
  -i "$input" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  "$output"
