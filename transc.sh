#!/usr/bin/env bash

set -u

API="http://localhost:8090"

for file in mass*.mp4; do
    [ -f "$file" ] || continue

    echo
    echo "========================================"
    echo "Submitting: $file"
    echo "========================================"

    # Submit job
    response=$(curl -sS \
        -X POST "$API/v1/jobs" \
        -F "file=@$file" \
        -F "language=fi" \
        -F "word_timestamps=true")

    echo "Response: $response"

    job_id=$(printf '%s' "$response" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')

    if [ -z "$job_id" ]; then
        echo "ERROR: Could not get job ID"
        continue
    fi

    echo "Job ID: $job_id"

    # Poll until completed
    while true; do
        status=$(curl -sS "$API/v1/jobs/$job_id")

        state=$(printf '%s' "$status" |
            sed -n 's/.*"status":"\([^"]*\)".*/\1/p')

        progress=$(printf '%s' "$status" |
            sed -n 's/.*"progress":\([^,}]*\).*/\1/p')

        printf '\rStatus: %-12s Progress: %s' "$state" "${progress:-?}"

        case "$state" in
            completed)
                echo
                break
                ;;

            failed|error)
                echo
                echo "ERROR: transcription failed:"
                echo "$status"
                break
                ;;
        esac

        sleep 5
    done

    # Only download VTT after successful completion
    if [ "$state" = "completed" ]; then
        output="${file%.mp4}.vtt"

        echo "Downloading: $output"

        curl -sS \
            "$API/v1/jobs/$job_id/result?format=vtt" \
            -o "$output"

        echo "Saved: $output"
    fi
done

echo
echo "All files processed."

