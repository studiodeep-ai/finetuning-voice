import os
import torch
import numpy as np
import soundfile as sf
import random
import re
import argparse
from safetensors.torch import load_file


from src.utils import setup_logger, trim_silence_with_vad
from src.config import TrainConfig
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("Chatterbox-Inference")


cfg = TrainConfig()

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
IS_TURBO = cfg.is_turbo
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir


def find_finetuned_weights(output_dir: str, is_turbo: bool) -> str:
    """
    Return the path to the best available finetuned weights:
    1. Final safetensors file (written at end of training)
    2. Latest checkpoint subfolder (written every N epochs during training)
    """
    filename = "t3_turbo_finetuned.safetensors" if is_turbo else "t3_finetuned.safetensors"
    final_path = os.path.join(output_dir, filename)
    if os.path.exists(final_path):
        return final_path

    # Fall back to latest checkpoint-NNNN directory
    checkpoints = sorted(
        [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ],
        key=lambda d: int(d.split("-")[-1]),
    ) if os.path.isdir(output_dir) else []

    if checkpoints:
        ckpt_path = os.path.join(output_dir, checkpoints[-1], filename)
        if os.path.exists(ckpt_path):
            return ckpt_path
        # HF Trainer saves as model.safetensors inside the checkpoint
        hf_path = os.path.join(output_dir, checkpoints[-1], "model.safetensors")
        if os.path.exists(hf_path):
            return hf_path

    return final_path  # return expected path so the error message is clear


def list_checkpoints(output_dir: str) -> list:
    """Return all checkpoint directories, sorted oldest → newest."""
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        [os.path.join(output_dir, d) for d in os.listdir(output_dir)
         if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))],
        key=lambda p: int(os.path.basename(p).split("-")[-1]),
    )


def load_finetuned_engine(device, weights_path: str = None, base_only: bool = False):
    """
    Loads the correct Chatterbox engine and replaces the T3 module with the
    fine-tuned version (or keeps the base model if base_only=True).
    """
    if base_only:
        logger.info("BASE MODEL MODE — loading pretrained weights only (no finetuning)")
    else:
        logger.info(f"Loading in {'TURBO' if IS_TURBO else 'NORMAL'} mode.")
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    EngineClass = ChatterboxTurboTTS if IS_TURBO else ChatterboxTTS

    tts_engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")

    if base_only:
        tts_engine.t3.to(device).eval()
        tts_engine.s3gen.to(device).eval()
        tts_engine.ve.to(device).eval()
        tts_engine.device = device
        return tts_engine

    # Configure New T3 Model
    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size

    new_t3 = T3(hp=t3_config)

    if IS_TURBO:
        logger.info("Turbo Mode: Removing 'wte' layer from new T3 model to match fine-tuned state.")
        if hasattr(new_t3.tfmr, "wte"):
            del new_t3.tfmr.wte

    weights = weights_path or find_finetuned_weights(OUTPUT_DIR, is_turbo=IS_TURBO)
    if os.path.exists(weights):
        logger.info(f"Loading fine-tuned weights: {weights}")
        state_dict = load_file(weights, device="cpu")
        new_t3.load_state_dict(state_dict, strict=True)
        logger.info("Fine-tuned weights loaded successfully.")
    else:
        logger.error(f"FATAL: Fine-tuned file not found at {weights}.")
        logger.error("Please ensure you have a trained model before running inference.")
        raise FileNotFoundError(weights)

    tts_engine.t3 = new_t3

    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device

    return tts_engine


def generate_sentence_audio(engine, text, prompt_path, **kwargs):
    """Generates audio for a single sentence and trims silence."""
    try:
        wav_tensor = engine.generate(text=text, audio_prompt_path=prompt_path, **kwargs)
        wav_np = wav_tensor.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
        return engine.sr, trimmed_wav
    except Exception as e:
        logger.error(f"Error generating sentence '{text[:30]}...': {e}")
        return 24000, np.zeros(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_params(args) -> dict:
    """Build generation parameters from CLI args."""
    if IS_TURBO:
        return {
            "temperature": args.temperature,
            "exaggeration": args.exaggeration,
            "repetition_penalty": args.repetition_penalty,
        }
    else:
        return {
            "temperature": args.temperature,
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
            "repetition_penalty": args.repetition_penalty,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox finetuned inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default finetuned inference
  python inference.py

  # Compare with base model (no finetuning)
  python inference.py --base-only

  # Try an earlier checkpoint
  python inference.py --list-checkpoints
  python inference.py --checkpoint chatterbox_output/checkpoint-50/model.safetensors

  # Tune for quality (if you hear repetition / short clips)
  python inference.py --repetition-penalty 1.5 --exaggeration 0.3

  # Custom text
  python inference.py --text "Hallo, dit is een test."
""",
    )

    # --- What to generate ---
    parser.add_argument("--text", default=cfg.inference_test_text,
                        help="Text to synthesise (default: config test text)")
    parser.add_argument("--output", default="./output_finetuned.wav",
                        help="Output WAV path (default: ./output_finetuned.wav)")

    # --- Which model to use ---
    parser.add_argument("--base-only", action="store_true",
                        help="Use the base pretrained model only (skips finetuned weights). "
                             "Useful for comparing finetuned vs. base quality.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a specific .safetensors checkpoint to load instead of the default.")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List all available checkpoints and exit.")

    # --- Reference audio ---
    parser.add_argument("--ref-audio", default=None,
                        help="Reference WAV for voice cloning (auto-detected from speaker_reference/ if omitted)")

    # --- Generation parameters ---
    # Defaults tuned to minimise early-EOS token repetition on Dutch:
    #   - higher repetition_penalty (1.5 vs 1.2) prevents immediate looping
    #   - lower exaggeration (0.3 vs 0.5) reduces guidance instability
    #   - temperature 0.8 is a stable default
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8). Try 0.7–1.0.")
    parser.add_argument("--exaggeration", type=float, default=0.3,
                        help="Voice expressiveness / guidance strength (default: 0.3). "
                             "Lower = more stable. Try 0.2–0.5.")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="Classifier-free guidance weight, standard mode only (default: 0.5). "
                             "Try 0.3–0.7.")
    parser.add_argument("--repetition-penalty", type=float, default=1.5,
                        help="Repetition penalty (default: 1.5). Increase to 2.0 if you hear loops.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # --- List checkpoints ---
    if args.list_checkpoints:
        ckpts = list_checkpoints(OUTPUT_DIR)
        if not ckpts:
            print(f"No checkpoints found in {OUTPUT_DIR}")
        else:
            print(f"Available checkpoints in {OUTPUT_DIR}:")
            for c in ckpts:
                # Show step number
                step = os.path.basename(c).split("-")[-1]
                print(f"  step {step:>6}  →  {c}")
            print(f"\nUsage: python inference.py --checkpoint <path>")
        return

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Inference running on: {device}")

    # --- Reference audio ---
    ref_dir = "speaker_reference"
    ref_files = sorted([
        f for f in os.listdir(ref_dir)
        if f.lower().endswith((".wav", ".mp3")) and not f.startswith(".")
    ]) if os.path.isdir(ref_dir) else []
    audio_prompt = args.ref_audio or (os.path.join(ref_dir, ref_files[0]) if ref_files else cfg.inference_prompt_path)

    # --- Load model ---
    engine = load_finetuned_engine(device, weights_path=args.checkpoint, base_only=args.base_only)

    # --- Build generation params ---
    params = build_params(args)
    logger.info(f"Generation params: {params}")

    # --- Synthesise ---
    sentences = re.split(r'(?<=[.?!])\s+', args.text.strip())
    sentences = [s for s in sentences if s.strip()]

    logger.info(f"Found {len(sentences)} sentences to synthesize.")

    all_chunks = []
    sample_rate = 24000

    set_seed(args.seed)

    for i, sent in enumerate(sentences):
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        sr, audio_chunk = generate_sentence_audio(engine, sent, audio_prompt, **params)

        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(args.output, final_audio, sample_rate)
        logger.info(f"Result saved to: {args.output}")
    else:
        logger.error("No audio was generated.")


if __name__ == "__main__":
    main()
