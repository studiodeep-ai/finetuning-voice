import os
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class FishSpeechTrainConfig:
    # --- Paths (absolute, relative to this file's parent) ---
    repo_dir: str             = str(SCRIPT_DIR / "repo")
    pretrained_model_dir: str = str(SCRIPT_DIR / "pretrained_models" / "openaudio-s1-mini")
    data_dir: str             = str(SCRIPT_DIR / "data")
    output_dir: str           = str(SCRIPT_DIR / "fish_speech_output")
    merged_model_dir: str     = str(SCRIPT_DIR / "merged_model")

    # --- Inference ---
    inference_prompt_path: str = str(SCRIPT_DIR / "speaker_reference" / "reference.wav")
    inference_test_text: str = (
        "Het Geluidshuis is een plek waar geluid centraal staat. "
        "Wij werken met de mooiste klanken uit de Nederlandse taal."
    )

    # --- Speaker ---
    speaker_name: str = field(default_factory=lambda: os.getenv("SPEAKER_NAME", "custom_voice"))

    # --- LoRA ---
    lora_config: str = "r_8_alpha_16"

    # --- Hyperparameters (overridable via environment variables) ---
    # batch_size: 4 is safe on 24GB GPU; try 8 on 40GB+
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "4")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))

    # Fish-speech uses step-based training (not epoch-based).
    # MAX_STEPS takes priority. If unset, uses NUM_EPOCHS × 50 (conservative steps/epoch
    # estimate for a small dataset of ~50 samples). Default: 1000 steps ≈ 20 epochs.
    max_steps: int = field(default_factory=lambda: int(
        os.getenv("MAX_STEPS", str(int(os.getenv("NUM_EPOCHS", "20")) * 50))
    ))

    # How often Lightning evaluates and saves checkpoints (in steps)
    val_check_interval: int = field(
        default_factory=lambda: int(os.getenv("VAL_CHECK_INTERVAL", "100"))
    )
