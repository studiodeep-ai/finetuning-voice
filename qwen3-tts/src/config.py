import os
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class Qwen3TrainConfig:
    # --- Paths ---
    model_dir: str = str(SCRIPT_DIR / "pretrained_models" / "model")
    tokenizer_dir: str = str(SCRIPT_DIR / "pretrained_models" / "tokenizer")
    wav_dir: str = str(SCRIPT_DIR / "FileBasedDataset")
    codes_path: str = str(SCRIPT_DIR / "FileBasedDataset" / "train_with_codes.jsonl")
    output_dir: str = str(SCRIPT_DIR / "qwen3_output")

    # --- Inference ---
    inference_prompt_path: str = str(SCRIPT_DIR / "speaker_reference" / "reference.wav")
    inference_test_text: str = (
        "Het Geluidshuis is een plek waar geluid centraal staat. "
        "Wij werken met de mooiste klanken uit de Nederlandse taal."
    )

    # --- Speaker ---
    speaker_name: str = field(default_factory=lambda: os.getenv("SPEAKER_NAME", "custom_voice"))

    # --- Hyperparameters (overridable via environment variables) ---
    # Qwen3-TTS is large (1.7B) — batch_size=1 is safe on 24GB, try 2 on 40GB+
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "1")))
    grad_accum: int = field(default_factory=lambda: int(os.getenv("GRAD_ACCUM", "4")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "10")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "2")))
    learning_rate: float = 2e-5

    # --- Audio codec batch size for prepare_dataset.py ---
    codec_batch_size: int = field(default_factory=lambda: int(os.getenv("CODEC_BATCH", "32")))
