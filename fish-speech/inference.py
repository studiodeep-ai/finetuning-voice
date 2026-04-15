"""
inference.py — Generate speech with a finetuned fish-speech model.

Loads the merged model from merged_model/ and the reference audio from
speaker_reference/, then synthesises inference_test_text.

Usage:
    python inference.py [--text "..."] [--output output.wav] [--ref-audio ref.wav]

The merged model must exist first — run train.py to create it.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import FishSpeechTrainConfig

cfg = FishSpeechTrainConfig()
REPO_DIR = Path(cfg.repo_dir)


def find_reference_audio(prompt_path: str) -> str:
    """Auto-detect reference audio: speaker_reference/ first, then fallback."""
    ref_dir = Path(cfg.inference_prompt_path).parent
    if ref_dir.exists():
        wavs = sorted(
            [f for f in ref_dir.iterdir()
             if f.suffix.lower() in {".wav", ".mp3"} and not f.name.startswith(".")]
        )
        if wavs:
            return str(wavs[0])
    return prompt_path


def main():
    parser = argparse.ArgumentParser(description="fish-speech inference with finetuned model")
    parser.add_argument("--text", default=cfg.inference_test_text,
                        help="Text to synthesize")
    parser.add_argument("--ref-audio", default=None,
                        help="Reference audio for voice cloning (auto-detected if omitted)")
    parser.add_argument("--output", default="output_finetuned.wav",
                        help="Output WAV file path")
    args = parser.parse_args()

    # --- Validate ---
    merged_dir = Path(cfg.merged_model_dir)
    if not merged_dir.exists() or not any(merged_dir.iterdir()):
        print(f"ERROR: Merged model not found at {merged_dir}")
        print("Run train.py first to train and merge the model.")
        sys.exit(1)

    if not REPO_DIR.exists():
        print(f"ERROR: fish-speech repo not found at {REPO_DIR}")
        print("Run setup.py first.")
        sys.exit(1)

    ref_audio = args.ref_audio or find_reference_audio(cfg.inference_prompt_path)
    if not Path(ref_audio).exists():
        print(f"WARNING: Reference audio not found at {ref_audio}")
        print("Continuing without voice reference (zero-shot).")
        ref_audio = None

    print("==========================================")
    print("  fish-speech Inference")
    print(f"  Model  : {merged_dir}")
    print(f"  Ref    : {ref_audio or '(none)'}")
    print(f"  Output : {args.output}")
    print("==========================================\n")
    print(f"Synthesising: {args.text[:80]}{'...' if len(args.text) > 80 else ''}\n")

    # --- Build inference command ---
    # fish-speech provides inference via: python fish_speech/inference.py
    # (or a CLI entry point — adjust if the fish-speech version differs)
    cmd = [
        sys.executable, "fish_speech/inference.py",
        "--model-dir", str(merged_dir),
        "--text", args.text,
        "--output", str(Path(args.output).resolve()),
    ]
    if ref_audio:
        cmd += ["--reference-audio", ref_audio]

    result = subprocess.run(cmd, cwd=str(REPO_DIR))
    if result.returncode != 0:
        print(f"\nERROR: Inference failed (exit code {result.returncode})")
        print("\nIf the command above is wrong for your fish-speech version, check:")
        print(f"  {REPO_DIR}/fish_speech/inference.py --help")
        sys.exit(result.returncode)

    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
