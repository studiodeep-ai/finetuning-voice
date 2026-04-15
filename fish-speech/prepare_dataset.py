"""
prepare_dataset.py — Build training data for fish-speech finetuning.

Four phases:
  1. Whisper segmentation → UUID.wav + UUID.lab in data/raw/<speaker>/
  2. Loudness normalization via `fap loudness-norm` → data/normalized/<speaker>/
  3. Semantic token extraction via fish-speech's extract_vq.py → adds .npy files
  4. Pack into protobuf format via fish-speech's build_dataset.py → data/protos/

Usage:
    python prepare_dataset.py --audio /path/to/audio [--language nl] [--speaker my_voice] [--clean]

Dependencies: openai-whisper, soundfile, fish-audio-preprocess (installed by setup.py)
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # expose common/

from common.audio_preparation import (
    TARGET_SR,
    find_audio_files,
    load_mono_16k,
    merge_short_segments,
    process_audio_file,
    save_chunk,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR / "repo"

MIN_DURATION = 3.0
MAX_DURATION = 15.0


def run_subprocess(cmd, cwd=None, description=""):
    """Run a subprocess and stream its output."""
    if description:
        print(f"  {description}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def phase1_whisper_segment(audio_dir: Path, speaker: str, language: str, clean: bool, min_dur: float, max_dur: float):
    """Whisper segmentation → data/raw/<speaker>/UUID.wav + UUID.lab"""
    out_dir = SCRIPT_DIR / "data" / "raw" / speaker
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for f in out_dir.glob("*.wav"):
            f.unlink()
        for f in out_dir.glob("*.lab"):
            f.unlink()
        print(f"  Cleaned data/raw/{speaker}/")

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"ERROR: no audio files found in {audio_dir}")
        sys.exit(1)

    print(f"  Found {len(audio_files)} audio file(s)")

    total_audio = 0.0
    total_chunks = 0

    for audio_file in audio_files:
        print(f"\n  Processing: {audio_file.name}")
        audio = load_mono_16k(str(audio_file))
        total_dur = len(audio) / TARGET_SR
        print(f"  Duration: {total_dur:.1f}s")

        segments = process_audio_file(audio_file, language, min_dur, max_dur)
        chunks = merge_short_segments(segments, min_dur, max_dur)
        print(f"  Segmented into {len(chunks)} chunks:")

        for chunk in chunks:
            # Use .lab extension (fish-speech format)
            save_chunk(audio, chunk["start"], chunk["end"], chunk["text"], out_dir, ext=".lab")
            total_audio += chunk["end"] - chunk["start"]
            total_chunks += 1

    print(f"\n  Phase 1 done: {total_chunks} pairs in data/raw/{speaker}/")
    print(f"  Total audio: {total_audio:.1f}s ({total_audio / 60:.1f} min)")

    if total_audio < 5 * 60:
        print(
            f"\n  WARNING: only {total_audio:.0f}s of audio. "
            f"Recommended: 30+ min for good voice similarity."
        )

    return total_chunks


def phase2_loudness_norm(speaker: str):
    """Loudness normalization via fap CLI."""
    src = SCRIPT_DIR / "data" / "raw" / speaker
    dst = SCRIPT_DIR / "data" / "normalized" / speaker
    dst.mkdir(parents=True, exist_ok=True)

    # fap is installed by fish-audio-preprocess (dep of fish-speech)
    run_subprocess(
        ["fap", "loudness-norm", str(src), str(dst), "--clean"],
        description="Normalizing loudness...",
    )
    print(f"  Phase 2 done: normalized audio in data/normalized/{speaker}/")


def phase3_extract_vq(speaker: str):
    """Extract semantic tokens (VQGAN codec)."""
    normalized_dir = SCRIPT_DIR / "data" / "normalized" / speaker
    codec_path = SCRIPT_DIR / "pretrained_models" / "openaudio-s1-mini" / "codec.pth"

    if not codec_path.exists():
        print(f"ERROR: codec not found at {codec_path}")
        print("Run: python3 setup.py")
        sys.exit(1)

    run_subprocess(
        [
            sys.executable, "tools/vqgan/extract_vq.py",
            str(normalized_dir),
            "--num-workers", "1",
            "--batch-size", "16",
            "--config-name", "modded_dac_vq",
            "--checkpoint-path", str(codec_path),
        ],
        cwd=REPO_DIR,
        description="Extracting semantic tokens...",
    )
    print("  Phase 3 done: semantic tokens extracted.")


def phase4_pack_dataset(speaker: str):
    """Pack into protobuf format for training."""
    normalized_dir = SCRIPT_DIR / "data" / "normalized"
    protos_dir = SCRIPT_DIR / "data" / "protos"
    protos_dir.mkdir(parents=True, exist_ok=True)

    run_subprocess(
        [
            sys.executable, "tools/llama/build_dataset.py",
            "--input", str(normalized_dir),
            "--output", str(protos_dir),
            "--text-extension", ".lab",
            "--num-workers", "4",
        ],
        cwd=REPO_DIR,
        description="Packing dataset into protobuf format...",
    )
    print(f"  Phase 4 done: training data packed → data/protos/")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fish-speech training dataset from audio files"
    )
    parser.add_argument("--audio", required=True, help="Path to folder with source audio files")
    parser.add_argument("--language", default="nl", help="Whisper language code (default: nl)")
    parser.add_argument("--speaker", default=None, help="Speaker name (default: SPEAKER_NAME env or 'custom_voice')")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION)
    parser.add_argument("--clean", action="store_true", help="Remove existing raw data before processing")
    args = parser.parse_args()

    audio_dir = Path(args.audio).resolve()
    if not audio_dir.is_dir():
        print(f"ERROR: audio folder not found: {audio_dir}")
        return 1

    import os
    speaker = args.speaker or os.getenv("SPEAKER_NAME", "custom_voice")

    if not REPO_DIR.exists():
        print("ERROR: fish-speech repo not found. Run: python3 setup.py")
        return 1

    print(f"Preparing fish-speech dataset for speaker: {speaker}")
    print(f"Audio source: {audio_dir}\n")

    print("=== Phase 1: Whisper segmentation ===")
    n = phase1_whisper_segment(audio_dir, speaker, args.language, args.clean, args.min_duration, args.max_duration)
    if n == 0:
        print("ERROR: no chunks produced. Check your audio files.")
        return 1

    print("\n=== Phase 2: Loudness normalization ===")
    phase2_loudness_norm(speaker)

    print("\n=== Phase 3: Semantic token extraction ===")
    phase3_extract_vq(speaker)

    print("\n=== Phase 4: Pack dataset ===")
    phase4_pack_dataset(speaker)

    print("\n==========================================")
    print("  Dataset preparation complete!")
    print(f"  Training data ready at: data/protos/")
    print("==========================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
