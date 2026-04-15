"""
setup.py — One-time setup for fish-speech finetuning.

What this does:
  1. Clones https://github.com/fishaudio/fish-speech → ./repo/
  2. Installs fish-speech Python package (torch-safe for RunPod)
  3. Downloads fishaudio/openaudio-s1-mini weights → ./pretrained_models/openaudio-s1-mini/

Run via:  python3 setup.py   (called automatically by install.sh)
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR / "repo"
PRETRAINED_DIR = SCRIPT_DIR / "pretrained_models" / "openaudio-s1-mini"

FISH_SPEECH_REPO = "https://github.com/fishaudio/fish-speech"
FISH_SPEECH_MODEL = "fishaudio/openaudio-s1-mini"


def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def step1_clone_repo():
    print("\n[1/3] Cloning fish-speech repository...")
    if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
        print(f"  repo/ already exists — skipping clone.")
        return
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", FISH_SPEECH_REPO, str(REPO_DIR)])
    print("  Cloned successfully.")


def step2_install_fish_speech():
    print("\n[2/3] Installing fish-speech package...")

    # Detect CUDA version for the correct extras tag
    cuda_tag = "cpu"
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; v=torch.version.cuda or ''; "
             "parts=v.split('.')[:2]; "
             "print('cu'+''.join(parts)) if v else print('cpu')"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            cuda_tag = result.stdout.strip()
    except Exception:
        pass
    print(f"  Detected compute tag: {cuda_tag}")

    # Detect whether torch is already installed (RunPod ships a GPU-specific build).
    # If so, install fish-speech WITHOUT its torch dependency to avoid downgrading.
    torch_installed = False
    try:
        subprocess.run(
            [sys.executable, "-c", "import torch"],
            check=True, capture_output=True,
        )
        torch_installed = True
    except subprocess.CalledProcessError:
        pass

    if torch_installed:
        print("  torch already installed — installing fish-speech without torch deps...")
        # Install the package itself (no ML deps)
        run([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR),
             "--no-deps", "--quiet"])
        # Install remaining non-torch dependencies that fish-speech needs
        extra_deps = [
            # Training infrastructure
            "lightning>=2.1.0",
            "hydra-core>=1.3.2",
            "pyrootutils>=1.0.4",
            # Audio processing
            "librosa>=0.10.1",
            "soundfile>=0.12.0",
            "resampy>=0.4.3",
            "pydub",
            "descript-audio-codec",   # codec/VQGAN weights loader
            # ML utilities
            "loralib>=0.1.2",
            "einops",
            "einx[torch]==0.2.2",
            "transformers",
            "vector_quantize_pytorch",
            "datasets==2.18.0",
            "pydantic==2.9.2",
            # Misc
            "natsort",
            "cachetools",
            "zstandard>=0.22.0",
            "ormsgpack",
            "tiktoken>=0.8.0",
            "tensorboard>=2.14.1",
            "silero-vad",
            "fish-audio-preprocess",  # provides the `fap` CLI for loudness normalization
        ]
        run([sys.executable, "-m", "pip", "install"] + extra_deps + ["--quiet"])
    else:
        print(f"  Installing fish-speech[{cuda_tag}]...")
        run([sys.executable, "-m", "pip", "install",
             "-e", f"{REPO_DIR}[{cuda_tag}]", "--quiet"])

    print("  fish-speech installed.")


def step3_download_model():
    print("\n[3/3] Downloading pretrained model weights...")
    if PRETRAINED_DIR.exists() and any(PRETRAINED_DIR.iterdir()):
        print(f"  pretrained_models/openaudio-s1-mini/ already downloaded — skipping.")
        return

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "huggingface-hub>=0.26.0", "--quiet"])
        from huggingface_hub import snapshot_download

    print(f"  Downloading {FISH_SPEECH_MODEL} (this may take several minutes)...")
    snapshot_download(
        repo_id=FISH_SPEECH_MODEL,
        local_dir=str(PRETRAINED_DIR),
    )
    print(f"  Model saved to: {PRETRAINED_DIR}")


def step4_create_tokenizer_config():
    """
    Create tokenizer_config.json so AutoTokenizer.from_pretrained works locally.

    openaudio-s1-mini ships without a tokenizer_config.json. It extends Qwen2's
    tiktoken vocabulary with fish-speech special tokens. Without this file,
    AutoTokenizer falls back to reading config.json, sees model_type='dual_ar'
    (a custom fish-speech architecture never registered with transformers), and
    raises ValueError. Providing tokenizer_config.json with tokenizer_class=
    Qwen2Tokenizer bypasses the AutoConfig lookup entirely.
    """
    import json
    import shutil

    config_file = PRETRAINED_DIR / "tokenizer_config.json"
    if config_file.exists():
        print("  tokenizer_config.json already exists — skipping.")
        return

    st_file = PRETRAINED_DIR / "special_tokens.json"
    if not st_file.exists():
        print("  WARNING: special_tokens.json not found — cannot create tokenizer config.")
        return

    special_tokens: dict = json.loads(st_file.read_text())

    # Qwen2Tokenizer loads its tiktoken vocab from a file named "qwen.tiktoken"
    src = PRETRAINED_DIR / "tokenizer.tiktoken"
    dst = PRETRAINED_DIR / "qwen.tiktoken"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print("  Copied tokenizer.tiktoken → qwen.tiktoken")

    # added_tokens_decoder tells HuggingFace to inject all fish-speech special
    # tokens with their exact IDs after the base Qwen2 vocab (0–151642).
    added_tokens_decoder = {
        str(token_id): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for token, token_id in special_tokens.items()
    }

    eos = "<|end_of_text|>" if "<|end_of_text|>" in special_tokens else "<|endoftext|>"
    pad = "<|pad|>" if "<|pad|>" in special_tokens else None

    tokenizer_config = {
        "added_tokens_decoder": added_tokens_decoder,
        "bos_token": None,
        "clean_up_tokenization_spaces": False,
        "eos_token": eos,
        "errors": "replace",
        "model_max_length": 32768,
        "pad_token": pad,
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
        "unk_token": None,
    }

    config_file.write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False))
    print(f"  Created tokenizer_config.json ({len(added_tokens_decoder)} special tokens)")


def main():
    print("==========================================")
    print("  fish-speech Setup")
    print(f"  Directory: {SCRIPT_DIR}")
    print("==========================================")

    step1_clone_repo()
    step2_install_fish_speech()
    step3_download_model()
    step4_create_tokenizer_config()

    print("\n==========================================")
    print("  Setup complete!")
    print("")
    print("  Next step:")
    print("    ./train.sh --model fish-speech --audio /path/to/audio")
    print("==========================================\n")


if __name__ == "__main__":
    main()
