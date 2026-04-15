import os
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # --- Paths ---
    model_dir: str = "./pretrained_models"
    csv_path: str = "./MyTTSDataset/metadata.csv"
    metadata_path: str = "./metadata.json"

    # File-based dataset (UUID.wav + UUID.txt pairs)
    wav_dir: str = "./FileBasedDataset"
    preprocessed_dir: str = "./FileBasedDataset/preprocess"

    output_dir: str = "./chatterbox_output"

    # --- Inference (used by inference.py and InferenceCallback) ---
    is_inference: bool = False
    # Point to the first WAV in speaker_reference/ (train.sh copies audio there)
    inference_prompt_path: str = "./speaker_reference/reference.wav"
    inference_test_text: str = (
        "Het Geluidshuis is een plek waar geluid centraal staat. "
        "Wij werken met de mooiste klanken uit de Nederlandse taal."
    )

    # --- Dataset format ---
    ljspeech: bool = False   # False = file-based pairs; True = LJSpeech CSV
    json_format: bool = False
    preprocess: bool = True  # Always re-preprocess on a fresh clone

    # --- Model variant ---
    is_turbo: bool = False   # False = Standard (Llama); True = Turbo (GPT-2)

    # --- Vocabulary ---
    # Standard mode: 2454 tokens (grapheme-based, 23 languages)
    # Turbo mode:    update after running setup.py (e.g. 52260)
    new_vocab_size: int = 52260 if is_turbo else 2454

    # --- Hyperparameters (overridable via environment variables) ---
    # On RunPod A100/A40: BATCH_SIZE=16, NUM_WORKERS=8
    # On RunPod RTX 3090: BATCH_SIZE=8, NUM_WORKERS=4
    # On local macOS MPS: BATCH_SIZE=4, NUM_WORKERS=2
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "4")))
    grad_accum: int = field(default_factory=lambda: int(os.getenv("GRAD_ACCUM", "8")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "50")))
    dataloader_num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "2")))

    learning_rate: float = 1e-5   # T3 is sensitive — keep low

    save_every_epochs: int = field(default_factory=lambda: int(os.getenv("SAVE_EVERY_EPOCHS", "10")))
    save_total_limit: int = 3
    val_split: float = 0.1   # Fraction of dataset held out for validation loss
    # Early stopping: how many evals with no val_loss improvement before stopping.
    # With save_every_epochs=10 (default), patience=3 means 30 epochs without improvement.
    early_stopping_patience: int = field(default_factory=lambda: int(os.getenv("EARLY_STOPPING_PATIENCE", "3")))

    # --- Sequence constraints ---
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_len: int = 256
    max_speech_len: int = 850    # Truncates very long audio segments
    prompt_duration: float = 3.0  # Seconds of reference audio used as prompt
