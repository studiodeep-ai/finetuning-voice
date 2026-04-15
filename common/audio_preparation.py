"""
audio_preparation.py — Shared audio segmentation utilities.

Used by both chatterbox/prepare_dataset.py and qwen3-tts/prepare_dataset.py
to avoid duplicating the Whisper transcription + segmentation pipeline.

Functions:
  load_mono_16k(audio_path)                           → np.ndarray
  transcribe_with_timestamps(audio_path, language)    → list[dict]
  merge_short_segments(segments, min_dur, max_dur)    → list[dict]
  save_chunk(audio, start, end, text, out_dir)        → str (UUID name)
  find_audio_files(folder)                            → list[Path]
  process_audio_file(audio_file, language, min_dur, max_dur) → list[dict]
"""

import json
import uuid
import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000  # Whisper and Chatterbox training expect 16 kHz


def load_mono_16k(audio_path: str) -> np.ndarray:
    """Load any audio file as mono 16kHz float32 via ffmpeg."""
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
    """Return Whisper segments list with start/end/text.

    Results are cached as <audio_path>.whisper.json next to the source file.
    Delete that file to force re-transcription.
    """
    import whisper

    cache_path = Path(audio_path).with_suffix(".whisper.json")

    if cache_path.exists():
        print(f"  Using cached transcription: {cache_path.name}")
        segments = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(segments)} cached segments.\n")
        return segments

    print("Loading Whisper model (medium) …")
    model = whisper.load_model("medium")
    print(f"Transcribing {audio_path} …")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=False,
        verbose=False,
    )
    segments = result["segments"]

    clean = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]
    cache_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Transcription cached to: {cache_path.name}")
    print(f"  Got {len(segments)} Whisper segments.\n")
    return clean


def merge_short_segments(segments, min_dur: float, max_dur: float):
    """Merge Whisper segments into chunks within [min_dur, max_dur] seconds."""
    chunks = []
    current_start = current_end = None
    current_texts = []

    for seg in segments:
        start, end, text = seg["start"], seg["end"], seg["text"].strip()
        if current_start is None:
            current_start, current_end, current_texts = start, end, [text]
        else:
            if end - current_start <= max_dur:
                current_end = end
                current_texts.append(text)
            else:
                if current_end - current_start >= min_dur:
                    chunks.append({
                        "start": current_start,
                        "end": current_end,
                        "text": " ".join(current_texts),
                    })
                current_start, current_end, current_texts = start, end, [text]

    if current_start is not None and current_end - current_start >= min_dur:
        chunks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts),
        })

    return chunks


def save_chunk(
    audio: np.ndarray,
    start: float,
    end: float,
    text: str,
    out_dir: Path,
    ext: str = ".txt",
) -> str:
    """Slice audio and write paired .wav + transcript to out_dir.

    Args:
        ext: Transcript file extension. Default ".txt"; use ".lab" for fish-speech.

    Returns the UUID stem name (without extension).
    """
    s, e = int(start * TARGET_SR), int(end * TARGET_SR)
    chunk = audio[s:e]
    name = str(uuid.uuid4())
    sf.write(str(out_dir / f"{name}.wav"), chunk, TARGET_SR, subtype="PCM_16")
    (out_dir / f"{name}{ext}").write_text(text, encoding="utf-8")
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
    Otherwise, run Whisper. Returns a list of segment dicts with start/end/text.
    """
    txt_file = audio_file.with_suffix(".txt")
    if txt_file.exists():
        print(f"  Using existing transcript: {txt_file.name}")
        transcript = txt_file.read_text(encoding="utf-8").strip()
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_file)],
            capture_output=True, text=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        return [{"start": 0.0, "end": duration, "text": transcript}]
    return transcribe_with_timestamps(str(audio_file), language)
