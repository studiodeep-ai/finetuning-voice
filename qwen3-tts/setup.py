"""
setup.py — Download Qwen3-TTS pretrained models from HuggingFace.

Run via: python3 setup.py
Or via:  ./install.sh --model qwen3-tts

Downloads:
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base   → pretrained_models/model/   (~3.5GB)
  - Qwen/Qwen3-TTS-Tokenizer-12Hz   → pretrained_models/tokenizer/ (~2GB)
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PRETRAINED_DIR = SCRIPT_DIR / "pretrained_models"
MODEL_DIR = PRETRAINED_DIR / "model"
TOKENIZER_DIR = PRETRAINED_DIR / "tokenizer"

MODELS = {
    "model": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": MODEL_DIR,
        "description": "Qwen3-TTS 1.7B base model (~3.5GB)",
    },
    "tokenizer": {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "local_dir": TOKENIZER_DIR,
        "description": "Qwen3-TTS 12Hz audio codec tokenizer (~2GB)",
    },
}


def download_model(name: str, repo_id: str, local_dir: Path, description: str):
    from huggingface_hub import snapshot_download

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  {name}: already downloaded, skipping.")
        return

    print(f"  Downloading {description}...")
    print(f"    repo: {repo_id}")
    print(f"    dest: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  {name}: done.")


def main():
    print("==========================================")
    print("  Qwen3-TTS: Downloading pretrained models")
    print("==========================================")
    print()

    try:
        import huggingface_hub
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in MODELS.items():
        download_model(name, cfg["repo_id"], cfg["local_dir"], cfg["description"])

    print()
    print("==========================================")
    print("  All models ready.")
    print(f"  Location: {PRETRAINED_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()
