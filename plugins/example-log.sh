#!/bin/bash
# Example plugin: Log transcript batch to stderr
# Receives JSON array on stdin, one batch at a time
#
# To enable: chmod +x plugins/example-log.sh
# To disable: chmod -x plugins/example-log.sh

# Read the JSON array from stdin and log it
BATCH=$(cat)
echo "[plugin] Received batch: ${#BATCH} bytes" >&2
