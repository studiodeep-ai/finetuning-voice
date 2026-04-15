"""
train.py — Fine-tune fish-speech (openaudio-s1-mini) on a custom voice.

Reads data/protos/ produced by prepare_dataset.py and runs LoRA finetuning
via the fish-speech training infrastructure (PyTorch Lightning + Hydra).

After training, automatically merges the LoRA adapters into the base model
and saves the merged weights to merged_model/.

Usage:
    python train.py          # uses src/config.py defaults + env var overrides
    MAX_STEPS=2000 python train.py
    NUM_EPOCHS=30 BATCH_SIZE=8 python train.py

Environment variables:
    SPEAKER_NAME          Speaker identifier (default: custom_voice)
    BATCH_SIZE            Batch size (default: 4)
    NUM_WORKERS           Dataloader workers (default: 4)
    MAX_STEPS             Training steps (default: NUM_EPOCHS × 50, or 1000)
    NUM_EPOCHS            Epochs shorthand — converted to steps (50 steps/epoch estimate)
    VAL_CHECK_INTERVAL    Steps between validation + checkpoint saves (default: 100)
"""

import re
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import FishSpeechTrainConfig


def find_latest_checkpoint(output_dir: str, speaker_name: str) -> Path:
    """Return the path to the most recent Lightning checkpoint."""
    base = Path(output_dir)

    # Lightning saves to: <output_dir>/<project>/checkpoints/step_XXXXXXXXXX.ckpt
    # Try project-specific location first, then scan broadly
    candidates = []
    for pattern in [
        base / speaker_name / "checkpoints" / "*.ckpt",
        base / "**" / "*.ckpt",
    ]:
        candidates.extend(base.glob(str(pattern.relative_to(base))))

    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt files found under {output_dir}. Training may have failed."
        )

    def step_number(p: Path) -> int:
        m = re.search(r"step[_-]?(\d+)", p.stem, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    return max(candidates, key=step_number)


def main():
    cfg = FishSpeechTrainConfig()

    print("==========================================")
    print("  fish-speech Finetuning")
    print(f"  Model  : {cfg.pretrained_model_dir}")
    print(f"  Output : {cfg.output_dir}")
    print(f"  Steps  : {cfg.max_steps}")
    print(f"  Batch  : {cfg.batch_size}")
    print(f"  Speaker: {cfg.speaker_name}")
    print("==========================================\n")

    # --- Validate prerequisites ---
    protos_dir = Path(cfg.data_dir) / "protos"
    if not protos_dir.exists() or not any(protos_dir.iterdir()):
        print(f"ERROR: Training data not found at {protos_dir}")
        print("Run: python3 prepare_dataset.py --audio /path/to/audio")
        sys.exit(1)

    repo_dir = Path(cfg.repo_dir)
    if not repo_dir.exists():
        print(f"ERROR: fish-speech repo not found at {repo_dir}")
        print("Run: python3 setup.py")
        sys.exit(1)

    pretrained_dir = Path(cfg.pretrained_model_dir)
    if not pretrained_dir.exists():
        print(f"ERROR: Pretrained model not found at {pretrained_dir}")
        print("Run: python3 setup.py")
        sys.exit(1)

    # --- Build Hydra override list ---
    # Keys verified against:
    #   fish-speech/repo/fish_speech/configs/text2semantic_finetune.yaml
    overrides = [
        f"project={cfg.speaker_name}",
        # Pretrained weights path (used by model loader)
        f"pretrained_ckpt_path={cfg.pretrained_model_dir}",
        # Tokenizer expects a directory, not the .tiktoken file itself
        f"tokenizer.model_path={cfg.pretrained_model_dir}",
        # LoRA config (config-group override — appends lora/r_8_alpha_16.yaml)
        f"+lora@model.model.lora_config={cfg.lora_config}",
        # Point proto_files to absolute path (default is relative "data/protos")
        f"train_dataset.proto_files=[{protos_dir}]",
        f"val_dataset.proto_files=[{protos_dir}]",
        # Training knobs
        f"trainer.max_steps={cfg.max_steps}",
        f"data.batch_size={cfg.batch_size}",
        f"data.num_workers={cfg.num_workers}",
        f"trainer.val_check_interval={cfg.val_check_interval}",
        f"trainer.default_root_dir={cfg.output_dir}",
    ]

    train_cmd = [sys.executable, "fish_speech/train.py", "--config-name", "text2semantic_finetune"] + overrides

    print("[1/2] Starting LoRA finetuning...")
    print(f"  Command: {' '.join(train_cmd[:6])} ...")
    print(f"  Running from: {repo_dir}\n")

    result = subprocess.run(train_cmd, cwd=str(repo_dir))
    if result.returncode != 0:
        print(f"\nERROR: Training failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print("\n[2/2] Merging LoRA adapters into base model...")

    try:
        latest_ckpt = find_latest_checkpoint(cfg.output_dir, cfg.speaker_name)
        print(f"  Latest checkpoint: {latest_ckpt}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    merged_dir = Path(cfg.merged_model_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    merge_cmd = [
        sys.executable, "tools/llama/merge_lora.py",
        "--lora-config", cfg.lora_config,
        "--base-weight", cfg.pretrained_model_dir,
        "--lora-weight", str(latest_ckpt),
        "--output", cfg.merged_model_dir,
    ]

    result = subprocess.run(merge_cmd, cwd=str(repo_dir))
    if result.returncode != 0:
        print(f"\nERROR: LoRA merge failed (exit code {result.returncode})")
        print(f"You can merge manually: {' '.join(merge_cmd)}")
        sys.exit(result.returncode)

    print(f"\n==========================================")
    print(f"  Training complete!")
    print(f"  LoRA checkpoints : {cfg.output_dir}")
    print(f"  Merged model     : {cfg.merged_model_dir}")
    print(f"")
    print(f"  To run inference:")
    print(f"    python3 inference.py")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
