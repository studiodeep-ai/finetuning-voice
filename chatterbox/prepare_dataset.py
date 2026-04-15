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
import uuid
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


DATASET_DIR = Path("FileBasedDataset")
MIN_DURATION = 3.0   # seconds — skip segments shorter than this
MAX_DURATION = 15.0  # seconds — split segments longer than this
TARGET_SR    = 16000  # Chatterbox training expects 16 kHz


def load_mono_16k(audio_path: str) -> np.ndarray:
    """Load any audio file as mono 16kHz float32."""
    import subprocess, tempfile, os

    # Use ffmpeg to convert to mono 16kHz WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", str(TARGET_SR),
            "-f", "wav", tmp.name,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audio, sr = sf.read(tmp.name, dtype="float32")
    os.unlink(tmp.name)
    assert sr == TARGET_SR
    return audio


def transcribe_with_timestamps(audio_path: str, language: str = "nl"):
    """Return Whisper segments list with start/end/text."""
    import whisper

    print(f"Loading Whisper model (medium) …")
    model = whisper.load_model("medium")

    print(f"Transcribing {audio_path} …")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=False,  # segment-level is enough
        verbose=False,
    )

    segments = result["segments"]
    print(f"  Got {len(segments)} Whisper segments.\n")
    return segments


def merge_short_segments(segments, min_dur: float, max_dur: float):
    """Merge Whisper segments into chunks within [min_dur, max_dur] seconds."""
    chunks = []
    current_start = None
    current_end   = None
    current_texts = []

    for seg in segments:
        start = seg["start"]
        end   = seg["end"]
        text  = seg["text"].strip()

        if current_start is None:
            current_start = start
            current_end   = end
            current_texts = [text]
        else:
            proposed_dur = end - current_start
            if proposed_dur <= max_dur:
                current_end = end
                current_texts.append(text)
            else:
                # Flush current chunk if long enough
                if (current_end - current_start) >= min_dur:
                    chunks.append({
                        "start": current_start,
                        "end":   current_end,
                        "text":  " ".join(current_texts),
                    })
                # Start fresh
                current_start = start
                current_end   = end
                current_texts = [text]

    # Flush last chunk
    if current_start is not None and (current_end - current_start) >= min_dur:
        chunks.append({
            "start": current_start,
            "end":   current_end,
            "text":  " ".join(current_texts),
        })

    return chunks


def save_chunk(audio: np.ndarray, start: float, end: float, text: str, out_dir: Path):
    """Slice audio and write paired .wav + .txt to out_dir."""
    s = int(start * TARGET_SR)
    e = int(end   * TARGET_SR)
    chunk = audio[s:e]

    name = str(uuid.uuid4())
    wav_path = out_dir / f"{name}.wav"
    txt_path = out_dir / f"{name}.txt"

    sf.write(str(wav_path), chunk, TARGET_SR, subtype="PCM_16")
    txt_path.write_text(text, encoding="utf-8")

    duration = len(chunk) / TARGET_SR
    print(f"  [{start:.1f}s – {end:.1f}s]  ({duration:.1f}s)  {text[:60]!r}")
    return name


def find_audio_files(folder: Path):
    """Return all audio files in the folder (non-recursive)."""
    exts = {".wav", ".WAV", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted([f for f in folder.iterdir() if f.suffix in exts])


def process_audio_file(audio_file: Path, language: str, min_dur: float, max_dur: float):
    """
    Process one audio file. If a matching .txt exists, use it as transcript.
    Otherwise, run Whisper. Returns a list of chunk dicts with start/end/text.
    """
    txt_file = audio_file.with_suffix(".txt")

    if txt_file.exists():
        # Pre-written transcript: treat the whole file as one unit,
        # then segment it by duration alone (split every max_dur seconds).
        print(f"  Using existing transcript: {txt_file.name}")
        transcript = txt_file.read_text(encoding="utf-8").strip()
        # Build a single pseudo-segment covering the whole file
        import subprocess, tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(audio_file)],
            capture_output=True,
        )
        import json, subprocess as sp
        probe = sp.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(audio_file)],
            capture_output=True, text=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        os.unlink(tmp.name)
        # Single segment — will be split in merge step if > max_dur
        segments = [{"start": 0.0, "end": duration, "text": transcript}]
    else:
        segments = transcribe_with_timestamps(str(audio_file), language)

    return segments


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

        # Load as mono 16kHz
        audio = load_mono_16k(str(audio_file))
        total_dur = len(audio) / TARGET_SR
        print(f"  Duration: {total_dur:.1f}s")

        # Get segments (Whisper or existing .txt)
        segments = process_audio_file(audio_file, args.language, args.min_duration, args.max_duration)

        # Merge into sensible chunks
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
