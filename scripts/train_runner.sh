#!/usr/bin/env bash
set -euo pipefail

# Default values
MAX_RETRIES=10
TRAIN_SCRIPT=""
CHECKPOINT_DIR=""
CONFIG_FILE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") \\
  --train-script-path PATH    Path to your training script (e.g. train_attn_head.py) \\
  --checkpoint-dir DIR        Directory containing run subfolders (e.g. ../runs/attn_pretrain) \\
  --config CONFIG             YAML config file to pass through \\
  [--max-retries N]           How many times to retry on crash (default: $MAX_RETRIES)
EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --train-script-path)
      TRAIN_SCRIPT="$2"; shift 2 ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    --config)
      CONFIG_FILE="$2"; shift 2 ;;
    --max-retries)
      MAX_RETRIES="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; usage ;;
  esac
done

# Validate
if [[ -z "$TRAIN_SCRIPT" || -z "$CHECKPOINT_DIR" || -z "$CONFIG_FILE" ]]; then
  echo "Missing required argument."
  usage
fi

echo "Training script: $TRAIN_SCRIPT"
echo "Checkpoint base dir: $CHECKPOINT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Max retries: $MAX_RETRIES"

for attempt in $(seq 1 "$MAX_RETRIES"); do
  echo "=== Attempt #$attempt ==="

  # Find all subdirectories under CHECKPOINT_DIR that contain latest.py,
  # sort them by modification time, and pick the first one.
  LATEST_DIR=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d \
      -exec test -e "{}/latest.pt" ';' \
      -printf '%T@ %p\n' \
    | sort -k1,1nr \
    | head -n1 \
    | cut -d' ' -f2-)

  if [[ -z "$LATEST_DIR" ]]; then
    echo "No directory with latest.pt checkpoint found. Starting training without checkpoint"
    LATEST_CKPT=""
  else 
    # Path to latest checkpoint file
    LATEST_CKPT="${LATEST_DIR%/}/latest.pt"
  fi

  echo "Using checkpoint: $LATEST_CKPT"
  
  # Run the training script, passing checkpoint and config
  if python "$TRAIN_SCRIPT" \
       --checkpoint="$LATEST_CKPT" \
       --config="$CONFIG_FILE"; then
    echo "✅ Success on attempt #$attempt"
    exit 0
  else
    echo "❌ Crash on attempt #$attempt; retrying in 5s…"
    sleep 5
  fi
done

echo "⚠️  All $MAX_RETRIES attempts failed; exiting."
exit 1
