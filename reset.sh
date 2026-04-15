#!/usr/bin/env bash
# reset.sh — Clean all generated files to prepare for a fresh training run.
#
# Usage:
#   ./reset.sh [--model chatterbox] [--keep-dataset] [--keep-output]
#
# What gets removed:
#   - FileBasedDataset/*.wav + *.txt  (segmented training pairs)
#   - FileBasedDataset/preprocess/    (cached .pt tensors)
#   - chatterbox_output/              (trained model checkpoints)
#   - speaker_reference/*             (copied reference audio)
#
# What is NOT removed by default:
#   - pretrained_models/              (3GB download — keep unless reinstalling)
#   - Your source audio folder        (never touched)
#
# Options:
#   --model        chatterbox | qwen3-tts  (default: chatterbox)
#   --keep-dataset keep FileBasedDataset pairs and tensors
#   --keep-output  keep chatterbox_output/ checkpoints

set -euo pipefail

MODEL="chatterbox"
KEEP_DATASET=false
KEEP_OUTPUT=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL="$2"; shift 2 ;;
    --keep-dataset) KEEP_DATASET=true; shift ;;
    --keep-output)  KEEP_OUTPUT=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/$MODEL"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory not found: $MODEL_DIR"
  exit 1
fi

echo "=========================================="
echo "  Resetting: $MODEL"
echo "=========================================="
echo ""

# --------------------------------------------------------------------------
# FileBasedDataset — segmented WAV/TXT pairs and preprocessed tensors
# --------------------------------------------------------------------------
if [[ "$KEEP_DATASET" == false ]]; then
  DATASET_DIR="$MODEL_DIR/FileBasedDataset"

  count=$(find "$DATASET_DIR" -maxdepth 1 \( -name "*.wav" -o -name "*.txt" \) 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$count" -gt 0 ]]; then
    find "$DATASET_DIR" -maxdepth 1 \( -name "*.wav" -o -name "*.txt" \) -delete
    echo "  Removed $count files from FileBasedDataset/"
  else
    echo "  FileBasedDataset/ already empty"
  fi

  PREPROCESS_DIR="$DATASET_DIR/preprocess"
  if [[ -d "$PREPROCESS_DIR" ]]; then
    rm -rf "$PREPROCESS_DIR"
    echo "  Removed FileBasedDataset/preprocess/"
  fi
else
  echo "  Skipping FileBasedDataset/ (--keep-dataset)"
fi

# --------------------------------------------------------------------------
# Training output — checkpoints and finetuned weights
# --------------------------------------------------------------------------
if [[ "$KEEP_OUTPUT" == false ]]; then
  OUTPUT_DIR="$MODEL_DIR/chatterbox_output"
  if [[ -d "$OUTPUT_DIR" ]]; then
    rm -rf "$OUTPUT_DIR"
    echo "  Removed chatterbox_output/"
  else
    echo "  chatterbox_output/ already empty"
  fi
else
  echo "  Skipping chatterbox_output/ (--keep-output)"
fi

# --------------------------------------------------------------------------
# Speaker reference audio
# --------------------------------------------------------------------------
REF_DIR="$MODEL_DIR/speaker_reference"
count=$(find "$REF_DIR" -maxdepth 1 \( -iname "*.wav" -o -iname "*.mp3" \) 2>/dev/null | wc -l | tr -d ' ')
if [[ "$count" -gt 0 ]]; then
  find "$REF_DIR" -maxdepth 1 \( -iname "*.wav" -o -iname "*.mp3" \) -delete
  echo "  Removed $count file(s) from speaker_reference/"
else
  echo "  speaker_reference/ already empty"
fi

echo ""
echo "=========================================="
echo "  Reset complete. Ready for a fresh run:"
echo ""
echo "    ./train.sh --model $MODEL --audio /path/to/audio"
echo "=========================================="
