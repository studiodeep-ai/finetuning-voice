"""
prepare_dataset.py — Segment reference audio into FileBasedDataset pairs.

Scans the input folder for WAV files. For each WAV:
  - If a matching .txt file exists → uses it as the transcript
  - Otherwise → runs Whisper medium to auto-transcribe

Then segments each audio file into 5–15 second chunks and saves
UUID-named .wav + .txt pairs into FileBasedDataset/.

Usage:
    python prepare_dataset.py --audio /path/to/audio/folder [--language nl] [--clean]

Dependencies (install via requirements.txt):
    openai-whisper soundfile numpy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # expose common/

from common.audio_preparation import (
    TARGET_SR,
    load_mono_16k,
    transcribe_with_timestamps,
    merge_short_segments,
    save_chunk,
    find_audio_files,
    process_audio_file,
)


DATASET_DIR = Path("FileBasedDataset")
MIN_DURATION = 3.0   # seconds — skip segments shorter than this
MAX_DURATION = 15.0  # seconds — split segments longer than this


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FileBasedDataset pairs from a folder of audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to folder containing WAV/MP3 audio files",
    )
    parser.add_argument(
        "--language",
        default="nl",
        help="Whisper language code, e.g. nl, en, fr (default: nl)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION,
        help=f"Minimum segment duration in seconds (default: {MIN_DURATION})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION,
        help=f"Maximum segment duration in seconds (default: {MAX_DURATION})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing FileBasedDataset contents before writing",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio).resolve()
    if not audio_dir.is_dir():
        print(f"ERROR: audio folder not found: {audio_dir}")
        return 1

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"ERROR: no audio files found in {audio_dir}")
        return 1

    print(f"Found {len(audio_files)} audio file(s) in {audio_dir}\n")

    # Prepare output dir
    DATASET_DIR.mkdir(exist_ok=True)
    if args.clean:
        for f in DATASET_DIR.glob("*.wav"):
            f.unlink()
        for f in DATASET_DIR.glob("*.txt"):
            f.unlink()
        print("Cleaned existing FileBasedDataset files.\n")

    total_training_audio = 0.0
    total_chunks = 0

    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")

        audio = load_mono_16k(str(audio_file))
        total_dur = len(audio) / TARGET_SR
        print(f"  Duration: {total_dur:.1f}s")

        segments = process_audio_file(audio_file, args.language, args.min_duration, args.max_duration)
        chunks = merge_short_segments(segments, args.min_duration, args.max_duration)
        print(f"  Segmented into {len(chunks)} chunks:")

        for chunk in chunks:
            save_chunk(audio, chunk["start"], chunk["end"], chunk["text"], DATASET_DIR)
            total_training_audio += chunk["end"] - chunk["start"]
            total_chunks += 1

        print()

    print(f"Done! Saved {total_chunks} pairs to {DATASET_DIR.resolve()}")
    print(f"Total training audio: {total_training_audio:.1f}s ({total_training_audio/60:.1f} min)")

    if total_training_audio < 5 * 60:
        print(
            f"\nWARNING: Only {total_training_audio:.0f}s of training audio."
            f" Recommended: 30+ minutes for good voice similarity."
            f"\nThe model will fine-tune but results may vary."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
