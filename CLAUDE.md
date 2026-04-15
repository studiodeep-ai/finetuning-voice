# Voice Finetuning Toolkit

## Purpose
This repo is a GPU-ready toolkit for finetuning TTS (text-to-speech) models on a custom voice. It is designed to:
- Run on a **RunPod GPU pod** (cloned fresh each session, trained, outputs synced back)
- Accept a folder of reference WAV audio as training data
- Support multiple TTS models in isolated subdirectories

## Entry Points
```bash
./install.sh --model chatterbox      # installs deps + downloads pretrained weights
./train.sh --model chatterbox --audio /path/to/audio [--batch 16] [--epochs 50]
```

## Repository Structure
```
finetuning-voice/
├── install.sh          # One-shot setup per model
├── train.sh            # Training entry point (calls prepare_dataset.py → train.py)
├── reset.sh            # Clean runtime dirs for a fresh training run
├── CLAUDE.md           # This file
├── README.md           # Usage guide
├── .gitignore
├── common/             # Shared utilities (imported by all models)
│   ├── audio_preparation.py  # Whisper segmentation + audio loading
│   └── training_status.py    # Overfitting status logic
├── chatterbox/         # Chatterbox TTS finetuning (HF Trainer, T3/Llama)
│   ├── requirements.txt
│   ├── setup.py            # Downloads pretrained_models/ from HuggingFace
│   ├── prepare_dataset.py  # Audio folder → FileBasedDataset/ via Whisper segmentation
│   ├── train.py            # Training script (T3 transformer, CUDA/MPS/CPU)
│   ├── inference.py        # Post-training inference test
│   ├── src/
│   │   ├── config.py       # TrainConfig — all hyperparameters + env var overrides
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── training_monitor.py  # Rich live dashboard callback
│   │   ├── training_ui.py       # Rich terminal dashboard
│   │   └── chatterbox_/    # Model internals (from gokhaneraslan/chatterbox-finetuning)
│   ├── FileBasedDataset/   # Runtime: UUID.wav + UUID.txt pairs
│   └── speaker_reference/  # Runtime: reference audio for inference
├── qwen3-tts/          # Qwen3-TTS finetuning (Accelerate, 1.7B)
│   ├── requirements.txt
│   ├── setup.py
│   ├── prepare_dataset.py  # Whisper segment + audio code extraction
│   ├── train.py
│   ├── inference.py
│   ├── src/
│   │   ├── config.py
│   │   └── dataset.py
│   └── speaker_reference/
└── fish-speech/        # fish-speech finetuning (Lightning + Hydra, LoRA, 0.5B)
    ├── requirements.txt
    ├── setup.py            # Clones fish-speech repo + downloads openaudio-s1-mini
    ├── prepare_dataset.py  # 4-phase: Whisper → loudness norm → VQ tokens → protobuf
    ├── train.py            # Subprocess wrapper: LoRA train + auto-merge
    ├── inference.py        # Generate speech from merged model
    ├── src/
    │   └── config.py       # FishSpeechTrainConfig — env var overrides
    ├── data/               # Runtime: raw/ normalized/ protos/ subdirs
    ├── pretrained_models/  # Runtime: openaudio-s1-mini/ weights
    ├── fish_speech_output/ # Runtime: Lightning LoRA checkpoints
    ├── merged_model/       # Runtime: merged weights ready for inference
    ├── repo/               # Runtime: cloned fishaudio/fish-speech GitHub repo
    └── speaker_reference/  # Runtime: reference audio
```

## How the Pipeline Works (Chatterbox)

1. **`prepare_dataset.py`** — takes `--audio /folder`, runs Whisper medium to transcribe + segment into 5–15s chunks, saves UUID-named `.wav`+`.txt` pairs to `FileBasedDataset/`
2. **`train.py`** — preprocesses dataset to `.pt` tensors (speaker embeddings + acoustic tokens), then trains the T3 transformer using HuggingFace Trainer
3. **`inference.py`** — loads the finetuned `t3_finetuned.safetensors` + reference audio, generates speech

## Configuration (`chatterbox/src/config.py`)
All hyperparameters are in `TrainConfig`. Key ones can be overridden via environment variables:

| Env var | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | `4` | Use `16` on 24GB GPU, `4` on 10GB |
| `GRAD_ACCUM` | `8` | Effective batch = BATCH × GRAD_ACCUM |
| `NUM_EPOCHS` | `50` | Low — dataset is small |
| `NUM_WORKERS` | `2` | Use `8` on RunPod |

## Dataset Format
`FileBasedDataset/` uses **file-based pairs** (not LJSpeech CSV):
- `<uuid>.wav` — audio segment, mono 16kHz
- `<uuid>.txt` — plain text transcription

`FileBasedDataset/preprocess/` holds cached `.pt` tensors — regenerated on every run, never committed to git.

## Git Policy
**Never commit:**
- `**/pretrained_models/` — 3GB, downloaded by setup.py
- `**/*.safetensors`, `**/*.pt` — model weights and preprocessed tensors
- `**/chatterbox_output/` — training outputs
- `**/preprocess/` — cached tensors

**Do commit:**
- All Python source code
- `requirements.txt`, `setup.py`
- `FileBasedDataset/*.wav` + `FileBasedDataset/*.txt` (small training samples, optional)

## RunPod Deployment Workflow
```bash
# 1. On RunPod pod (PyTorch 2.6 + CUDA 12.4 template):
git clone https://github.com/<you>/finetuning-voice.git
cd finetuning-voice
./install.sh --model chatterbox   # ~5-10 min first time (RTX 5090: use PyTorch 2.8 template)

# 2. Upload audio from your local machine (separate terminal):
rsync -avz -e "ssh -p <port>" /path/to/audio/ root@<pod-ip>:/workspace/finetuning-voice/audio/

# 3. Train:
./train.sh --model chatterbox --audio /workspace/finetuning-voice/audio --batch 16

# 4. Download results to local machine:
rsync -avz -e "ssh -p <port>" root@<pod-ip>:/workspace/finetuning-voice/chatterbox/chatterbox_output/ ./outputs/
```

## How the Pipeline Works (fish-speech)

1. **`prepare_dataset.py`** — 4-phase pipeline:
   - Phase 1: Whisper segments audio → `data/raw/<speaker>/UUID.wav` + `UUID.lab`
   - Phase 2: `fap loudness-norm` normalises audio → `data/normalized/<speaker>/`
   - Phase 3: `extract_vq.py` extracts VQGAN semantic tokens (adds files alongside audio)
   - Phase 4: `build_dataset.py` packs into protobuf → `data/protos/`
2. **`train.py`** — calls `fish_speech/train.py` via subprocess with Hydra overrides; after training automatically runs `merge_lora.py` → `merged_model/`
3. **`inference.py`** — calls fish-speech inference with merged model + reference audio

## Configuration (fish-speech/src/config.py)
| Env var | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | `4` | Use `8` on 40GB+ GPU |
| `NUM_WORKERS` | `4` | Dataloader workers |
| `MAX_STEPS` | `1000` | Training steps (takes priority over NUM_EPOCHS) |
| `NUM_EPOCHS` | `20` | Converted to steps: `NUM_EPOCHS × 50` |
| `SPEAKER_NAME` | `custom_voice` | Speaker identifier |
| `VAL_CHECK_INTERVAL` | `100` | Steps between validation + checkpoint saves |

**Note:** fish-speech uses `openaudio-s1-mini` (0.5B, CC-BY-NC-SA-4.0). Do NOT use S2-Pro for finetuning — it is RL-trained and the docs warn against it.

## Adding a New Model
1. Create `<model>/` directory
2. Add: `requirements.txt`, `setup.py`, `prepare_dataset.py`, `train.py`, `inference.py`
3. `train.sh` calls `prepare_dataset.py --audio <path> --clean` then `train.py` — match this interface
4. Use env vars `BATCH_SIZE`, `NUM_EPOCHS`, `NUM_WORKERS`, `SPEAKER_NAME` for config overrides
5. `setup.py` must create `pretrained_models/` (train.sh checks for it)
6. Update available models list in `install.sh`, `train.sh`, `reset.sh`
7. Add model section to this CLAUDE.md

## Known Issues
- **macOS MPS**: Chatterbox training crashes in the T3 transformer (MPS incompatibility). Use CUDA (RunPod) or CPU (very slow). Preprocessing works fine on macOS.
- **Small dataset**: ~78s (6 segments) of reference audio will fine-tune but voice similarity may be limited. Recommended: 30+ minutes of clean audio.
- **fish-speech Hydra overrides**: The exact config keys (`data.train.filelist`, `trainer.default_root_dir`, etc.) must match `fish-speech/repo/fish_speech/configs/text2semantic_finetune.yaml`. Verify after cloning the repo if training fails to start.
