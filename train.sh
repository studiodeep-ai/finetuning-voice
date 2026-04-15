#!/usr/bin/env bash
# train.sh — Finetune a TTS model on your reference voice.
#
# Usage:
#   ./train.sh --model chatterbox --audio /path/to/audio/folder [options]
#
# Options:
#   --model    chatterbox | qwen3-tts          (required)
#   --audio    path to folder with WAV files   (required)
#   --epochs   number of training epochs       (default: model's config default)
#   --batch    batch size per device           (default: model's config default)
#
# The audio folder should contain:
#   - One or more WAV files (any sample rate, mono or stereo)
#   - Optionally matching .txt transcription files (same basename)
#     If no .txt files found, Whisper transcribes automatically.
#
# Examples:
#   ./train.sh --model chatterbox --audio ./audio/evam
#   ./train.sh --model chatterbox --audio ./audio/evam --batch 16 --epochs 100
#   BATCH_SIZE=16 NUM_EPOCHS=100 ./train.sh --model chatterbox --audio ./audio/evam

set -euo pipefail

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
MODEL=""
AUDIO=""
EPOCHS=""
BATCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)  MODEL="$2";  shift 2 ;;
    --audio)  AUDIO="$2";  shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch)  BATCH="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$AUDIO" ]]; then
  echo "Usage: $0 --model <model> --audio <audio-folder> [--epochs N] [--batch N]"
  echo ""
  echo "Available models: chatterbox, qwen3-tts"
  exit 1
fi

AUDIO="$(cd "$AUDIO" && pwd)"   # resolve to absolute path

if [[ ! -d "$AUDIO" ]]; then
  echo "ERROR: audio directory not found: $AUDIO"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/$MODEL"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory not found: $MODEL_DIR"
  echo "Run: ./install.sh --model $MODEL"
  exit 1
fi

if [[ ! -d "$MODEL_DIR/pretrained_models" ]]; then
  echo "ERROR: pretrained models not found. Run: ./install.sh --model $MODEL"
  exit 1
fi

echo "=========================================="
echo "  Training: $MODEL"
echo "  Audio   : $AUDIO"
echo "  Device  : $(python3 -c "import torch; d='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'); print(d)" 2>/dev/null || echo 'unknown')"
echo "=========================================="

# --------------------------------------------------------------------------
# Export optional overrides (picked up by src/config.py)
# --------------------------------------------------------------------------
[[ -n "$BATCH" ]]  && export BATCH_SIZE="$BATCH"
[[ -n "$EPOCHS" ]] && export NUM_EPOCHS="$EPOCHS"

# --------------------------------------------------------------------------
# Run model-specific training pipeline
# --------------------------------------------------------------------------
cd "$MODEL_DIR"

echo ""
echo "[1/3] Preparing dataset from: $AUDIO"
python3 prepare_dataset.py --audio "$AUDIO" --clean

echo ""
echo "[2/3] Copying reference audio to speaker_reference/"
mkdir -p speaker_reference
# Copy WAV files (both .wav and .WAV extensions)
find "$AUDIO" -maxdepth 1 \( -iname "*.wav" -o -iname "*.mp3" \) -exec cp {} speaker_reference/ \; 2>/dev/null || true
REFCOUNT=$(find speaker_reference -maxdepth 1 \( -iname "*.wav" -o -iname "*.mp3" \) | wc -l | tr -d ' ')
echo "  $REFCOUNT reference file(s) ready."

echo ""
echo "[3/3] Starting training..."
python3 train.py

echo ""
echo "=========================================="
echo "  Training complete!"
echo "  Output: $MODEL_DIR/chatterbox_output/"
echo ""
echo "  To run inference:"
echo "    cd $MODEL_DIR && python3 inference.py"
echo "=========================================="
