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
#   --model        chatterbox | qwen3-tts | fish-speech  (default: chatterbox)
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
# Dataset — segmented audio pairs and intermediate files
# --------------------------------------------------------------------------
if [[ "$KEEP_DATASET" == false ]]; then

  if [[ "$MODEL" == "fish-speech" ]]; then
    # fish-speech uses data/ instead of FileBasedDataset/
    DATA_DIR="$MODEL_DIR/data"
    if [[ -d "$DATA_DIR" ]]; then
      rm -rf "$DATA_DIR"
      echo "  Removed data/ (raw, normalized, protos)"
    else
      echo "  data/ already empty"
    fi
  else
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

    CODES_JSONL="$DATASET_DIR/train_with_codes.jsonl"
    if [[ -f "$CODES_JSONL" ]]; then
      rm -f "$CODES_JSONL"
      echo "  Removed FileBasedDataset/train_with_codes.jsonl"
    fi
  fi

else
  echo "  Skipping dataset dir (--keep-dataset)"
fi

# --------------------------------------------------------------------------
# Training output — checkpoints and finetuned weights
# --------------------------------------------------------------------------
if [[ "$KEEP_OUTPUT" == false ]]; then
  # Detect output dir name per model
  case "$MODEL" in
    chatterbox)   OUTPUT_DIRNAME="chatterbox_output" ;;
    qwen3-tts)    OUTPUT_DIRNAME="qwen3_output" ;;
    fish-speech)  OUTPUT_DIRNAME="fish_speech_output" ;;
    *)            OUTPUT_DIRNAME="${MODEL}_output" ;;
  esac
  OUTPUT_DIR="$MODEL_DIR/$OUTPUT_DIRNAME"
  if [[ -d "$OUTPUT_DIR" ]]; then
    rm -rf "$OUTPUT_DIR"
    echo "  Removed $OUTPUT_DIRNAME/"
  else
    echo "  $OUTPUT_DIRNAME/ already empty"
  fi

  # fish-speech also has a merged_model/ directory
  if [[ "$MODEL" == "fish-speech" ]]; then
    MERGED_DIR="$MODEL_DIR/merged_model"
    if [[ -d "$MERGED_DIR" ]]; then
      rm -rf "$MERGED_DIR"
      echo "  Removed merged_model/"
    fi
  fi
else
  echo "  Skipping output dir (--keep-output)"
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
