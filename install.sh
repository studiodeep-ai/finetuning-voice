#!/usr/bin/env bash
# install.sh — Set up a voice finetuning model on a fresh machine (RunPod or local).
#
# Usage:
#   ./install.sh --model chatterbox
#   ./install.sh chatterbox          (shorthand)
#
# What it does:
#   1. Checks system dependencies (Python 3.8+, ffmpeg)
#   2. Installs Python packages from <model>/requirements.txt
#   3. Downloads pretrained model weights via <model>/setup.py

set -euo pipefail

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
MODEL=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    -*)      echo "Unknown flag: $1"; exit 1 ;;
    *)       MODEL="$1"; shift ;;   # positional
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 --model <model>"
  echo "Available models: chatterbox, qwen3-tts"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/$MODEL"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory not found: $MODEL_DIR"
  exit 1
fi

echo "=========================================="
echo "  Installing: $MODEL"
echo "  Directory : $MODEL_DIR"
echo "=========================================="

# --------------------------------------------------------------------------
# 1. System dependencies
# --------------------------------------------------------------------------
echo ""
echo "[1/3] Checking system dependencies..."

# Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.8+ first."
  exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYTHON_VERSION"

# ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "  ffmpeg not found — installing..."
  if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y ffmpeg
  elif command -v brew &>/dev/null; then
    brew install ffmpeg
  else
    echo "  WARNING: ffmpeg not found and auto-install not supported on this OS."
    echo "  Install manually: https://ffmpeg.org/download.html"
  fi
else
  echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
fi

# --------------------------------------------------------------------------
# 2. Python dependencies
# --------------------------------------------------------------------------
echo ""
echo "[2/3] Installing Python packages..."
pip install -r "$MODEL_DIR/requirements.txt" --quiet

# --------------------------------------------------------------------------
# 3. Download pretrained models
# --------------------------------------------------------------------------
echo ""
echo "[3/3] Downloading pretrained model weights..."
cd "$MODEL_DIR"
python3 setup.py
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "  Done! $MODEL is ready."
echo ""
echo "  Next step:"
echo "    ./train.sh --model $MODEL --audio /path/to/audio"
echo "=========================================="
