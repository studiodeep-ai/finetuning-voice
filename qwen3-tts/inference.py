"""
inference.py — Test a finetuned Qwen3-TTS checkpoint.

Loads the most recent checkpoint from qwen3_output/, uses the reference
audio in speaker_reference/ as the voice prompt, and generates speech.

Usage:
    python inference.py
    python inference.py --checkpoint qwen3_output/checkpoint-epoch-2 --text "Hallo wereld"
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Qwen3TrainConfig


def find_latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-epoch-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def main():
    cfg = Qwen3TrainConfig()

    parser = argparse.ArgumentParser(description="Run inference with a finetuned Qwen3-TTS checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint dir (default: latest)")
    parser.add_argument("--ref-audio", default=cfg.inference_prompt_path, help="Reference WAV for voice cloning")
    parser.add_argument("--text", default=cfg.inference_test_text, help="Text to synthesize")
    parser.add_argument("--output", default="output_finetuned.wav", help="Output WAV file")
    parser.add_argument("--speaker", default=cfg.speaker_name, help="Speaker name (must match training)")
    args = parser.parse_args()

    # Find checkpoint
    checkpoint = args.checkpoint or find_latest_checkpoint(cfg.output_dir)
    if checkpoint is None:
        print(f"ERROR: No checkpoints found in {cfg.output_dir}")
        print("Run train.py first.")
        sys.exit(1)

    print(f"Checkpoint : {checkpoint}")
    print(f"Reference  : {args.ref_audio}")
    print(f"Text       : {args.text[:80]}")
    print(f"Speaker    : {args.speaker}")
    print()

    if not os.path.exists(args.ref_audio):
        print(f"ERROR: Reference audio not found: {args.ref_audio}")
        sys.exit(1)

    # --- Load model ---
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint on {device} …")

    model = Qwen3TTSModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    # --- Generate ---
    print("Generating speech …")
    with torch.no_grad():
        result = model.generate(
            text=args.text,
            ref_audio=args.ref_audio,
            speaker_name=args.speaker,
        )

    # --- Save ---
    import torchaudio
    torchaudio.save(args.output, result.audio.cpu(), result.sample_rate)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
