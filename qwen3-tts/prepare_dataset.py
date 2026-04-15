"""
prepare_dataset.py — Build training data for Qwen3-TTS finetuning.

Two phases:
  1. Segment + transcribe audio → UUID.wav + UUID.txt in FileBasedDataset/
     (identical to chatterbox pipeline; uses .whisper.json cache)
  2. Extract 16-layer audio codes via Qwen3TTSTokenizer → train_with_codes.jsonl

Usage:
    python prepare_dataset.py --audio /path/to/audio [--language nl] [--clean]

Dependencies: openai-whisper, soundfile, qwen-tts
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

DATASET_DIR = Path("FileBasedDataset")
CODES_JSONL = DATASET_DIR / "train_with_codes.jsonl"
MIN_DURATION = 3.0
MAX_DURATION = 15.0
TARGET_SR = 16000  # Whisper segmentation SR
QWEN_SR = 24000   # Qwen3-TTS expects 24kHz for codec extraction


# ---------------------------------------------------------------------------
# Phase 1: Whisper segmentation (same as chatterbox)
# ---------------------------------------------------------------------------

def load_mono_16k(audio_path: str) -> np.ndarray:
    import subprocess, tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", str(TARGET_SR), "-f", "wav", tmp.name],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    audio, sr = sf.read(tmp.name, dtype="float32")
    os.unlink(tmp.name)
    assert sr == TARGET_SR
    return audio


def transcribe_with_timestamps(audio_path: str, language: str = "nl"):
    """Return Whisper segments, cached as .whisper.json next to the source file."""
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
    result = model.transcribe(audio_path, language=language, word_timestamps=False, verbose=False)
    segments = result["segments"]

    clean = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]
    cache_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Transcription cached to: {cache_path.name}")
    print(f"  Got {len(segments)} Whisper segments.\n")
    return clean


def merge_short_segments(segments, min_dur: float, max_dur: float):
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
                    chunks.append({"start": current_start, "end": current_end, "text": " ".join(current_texts)})
                current_start, current_end, current_texts = start, end, [text]

    if current_start is not None and current_end - current_start >= min_dur:
        chunks.append({"start": current_start, "end": current_end, "text": " ".join(current_texts)})

    return chunks


def save_chunk(audio: np.ndarray, start: float, end: float, text: str, out_dir: Path):
    s, e = int(start * TARGET_SR), int(end * TARGET_SR)
    chunk = audio[s:e]
    name = str(uuid.uuid4())
    sf.write(str(out_dir / f"{name}.wav"), chunk, TARGET_SR, subtype="PCM_16")
    (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    duration = len(chunk) / TARGET_SR
    print(f"  [{start:.1f}s – {end:.1f}s]  ({duration:.1f}s)  {text[:60]!r}")
    return name


def find_audio_files(folder: Path):
    exts = {".wav", ".WAV", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted([f for f in folder.iterdir() if f.suffix in exts])


def process_audio_file(audio_file: Path, language: str, min_dur: float, max_dur: float):
    txt_file = audio_file.with_suffix(".txt")
    if txt_file.exists():
        print(f"  Using existing transcript: {txt_file.name}")
        transcript = txt_file.read_text(encoding="utf-8").strip()
        import json as _json, subprocess as sp
        probe = sp.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_file)],
            capture_output=True, text=True,
        )
        duration = float(_json.loads(probe.stdout)["format"]["duration"])
        return [{"start": 0.0, "end": duration, "text": transcript}]
    return transcribe_with_timestamps(str(audio_file), language)


def run_phase1(audio_dir: Path, language: str, clean: bool, min_dur: float, max_dur: float):
    """Segment + transcribe → FileBasedDataset/"""
    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"ERROR: no audio files found in {audio_dir}")
        return 1

    print(f"Found {len(audio_files)} audio file(s) in {audio_dir}\n")

    DATASET_DIR.mkdir(exist_ok=True)
    if clean:
        for f in DATASET_DIR.glob("*.wav"):
            f.unlink()
        for f in DATASET_DIR.glob("*.txt"):
            f.unlink()
        if CODES_JSONL.exists():
            CODES_JSONL.unlink()
        print("Cleaned existing FileBasedDataset files.\n")

    total_audio = 0.0
    total_chunks = 0

    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        audio = load_mono_16k(str(audio_file))
        total_dur = len(audio) / TARGET_SR
        print(f"  Duration: {total_dur:.1f}s")

        segments = process_audio_file(audio_file, language, min_dur, max_dur)
        chunks = merge_short_segments(segments, min_dur, max_dur)
        print(f"  Segmented into {len(chunks)} chunks:")

        for chunk in chunks:
            save_chunk(audio, chunk["start"], chunk["end"], chunk["text"], DATASET_DIR)
            total_audio += chunk["end"] - chunk["start"]
            total_chunks += 1
        print()

    print(f"Phase 1 done! Saved {total_chunks} pairs to {DATASET_DIR.resolve()}")
    print(f"Total training audio: {total_audio:.1f}s ({total_audio / 60:.1f} min)\n")

    if total_audio < 5 * 60:
        print(f"WARNING: Only {total_audio:.0f}s of training audio. Recommended: 30+ minutes.\n")

    return 0


# ---------------------------------------------------------------------------
# Phase 2: Audio code extraction via Qwen3TTSTokenizer
# ---------------------------------------------------------------------------

def load_mono_24k(wav_path: str) -> str:
    """Convert wav to 24kHz mono temp file, return path."""
    import subprocess, tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-ac", "1", "-ar", str(QWEN_SR), "-f", "wav", tmp.name],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmp.name


def run_phase2(tokenizer_dir: str, ref_audio: str, codec_batch_size: int = 32):
    """Extract audio codes → train_with_codes.jsonl"""
    from qwen_tts import Qwen3TTSTokenizer
    from tqdm import tqdm

    wav_files = sorted(DATASET_DIR.glob("*.wav"))
    if not wav_files:
        print("ERROR: No WAV files found in FileBasedDataset/. Run Phase 1 first.")
        return 1

    print(f"Loading Qwen3TTSTokenizer from {tokenizer_dir} …")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_dir, device_map=device)

    # Build data list with text + ref_audio
    data_list = []
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  WARNING: no transcript for {wav_path.name}, skipping.")
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        data_list.append({"audio": str(wav_path), "text": text, "ref_audio": ref_audio})

    print(f"Extracting audio codes for {len(data_list)} samples (batch={codec_batch_size}) …")

    final_lines = []
    batch_items = []
    batch_audios_24k = []

    def flush_batch():
        tmp_paths = []
        for item in batch_items:
            tmp = load_mono_24k(item["audio"])
            tmp_paths.append(tmp)
            batch_audios_24k.append(tmp)

        enc = tokenizer.encode(batch_audios_24k)
        for code, item in zip(enc.audio_codes, batch_items):
            item["audio_codes"] = code.cpu().tolist()
            final_lines.append(item)

        # Cleanup temp files
        import os
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        batch_items.clear()
        batch_audios_24k.clear()

    for item in tqdm(data_list, desc="Encoding audio"):
        batch_items.append(item)
        if len(batch_items) >= codec_batch_size:
            flush_batch()

    if batch_items:
        flush_batch()

    # Write JSONL
    CODES_JSONL.parent.mkdir(exist_ok=True)
    with open(CODES_JSONL, "w", encoding="utf-8") as f:
        for item in final_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Phase 2 done! {len(final_lines)} samples written to {CODES_JSONL.resolve()}\n")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen3-TTS training dataset")
    parser.add_argument("--audio", required=True, help="Folder with source audio files")
    parser.add_argument("--language", default="nl", help="Whisper language code (default: nl)")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION)
    parser.add_argument("--clean", action="store_true", help="Remove existing dataset before processing")
    parser.add_argument("--tokenizer-dir", default=None, help="Path to Qwen3TTSTokenizer (default: pretrained_models/tokenizer)")
    parser.add_argument("--codec-batch", type=int, default=32, help="Batch size for audio code extraction")
    args = parser.parse_args()

    audio_dir = Path(args.audio).resolve()
    if not audio_dir.is_dir():
        print(f"ERROR: audio folder not found: {audio_dir}")
        return 1

    # Determine reference audio (first file in audio dir, or speaker_reference/)
    ref_audio = None
    speaker_ref_dir = Path("speaker_reference")
    if speaker_ref_dir.exists():
        refs = sorted(speaker_ref_dir.glob("*.wav")) + sorted(speaker_ref_dir.glob("*.WAV"))
        if refs:
            ref_audio = str(refs[0])

    if ref_audio is None:
        audio_files = find_audio_files(audio_dir)
        if audio_files:
            ref_audio = str(audio_files[0])

    if ref_audio is None:
        print("ERROR: No reference audio found. Provide audio in --audio or speaker_reference/")
        return 1

    print(f"Reference audio for speaker embedding: {ref_audio}\n")

    # Phase 1: segment + transcribe
    rc = run_phase1(audio_dir, args.language, args.clean, args.min_duration, args.max_duration)
    if rc != 0:
        return rc

    # Phase 2: audio code extraction
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        script_dir = Path(__file__).parent.resolve()
        tokenizer_dir = str(script_dir / "pretrained_models" / "tokenizer")

    rc = run_phase2(tokenizer_dir, ref_audio, args.codec_batch)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
