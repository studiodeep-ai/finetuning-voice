# Voice Finetuning Toolkit

Finetune TTS models on a custom voice. Designed for **RunPod GPU pods** — clone, install, train, done.

Currently supported: **Chatterbox TTS** (Standard mode, Llama-based, 23 languages including Dutch)  
Coming soon: **Qwen3-TTS**

---

## Quick Start (RunPod)

```bash
# 1. Clone and install
git clone https://github.com/<you>/fintuning-voice.git
cd fintuning-voice
./install.sh --model chatterbox

# 2. Upload your voice audio (from local machine)
rsync -avz -e "ssh -p <PORT>" /path/to/audio/ root@<POD-IP>:/workspace/fintuning-voice/audio/

# 3. Train
./train.sh --model chatterbox --audio /workspace/fintuning-voice/audio --batch 16

# 4. Download trained model (from local machine)
rsync -avz -e "ssh -p <PORT>" root@<POD-IP>:/workspace/fintuning-voice/chatterbox/chatterbox_output/ ./outputs/
```

---

## Requirements

- Python 3.8+
- ffmpeg
- CUDA GPU recommended (RTX 3090 / A40 / A100 for batch_size=16)
- RunPod template: **PyTorch 2.6.0 / CUDA 12.4**

---

## Audio Input

The `--audio` folder can contain:
- One or more `.wav` files (any sample rate, mono or stereo)
- Optionally `.txt` files with matching names (pre-written transcriptions)

If no `.txt` files are found, **Whisper medium** transcribes automatically.

**Recommended:** 30+ minutes of clean, noise-free speech for best results.  
Minimum: 10 seconds (usable but limited voice similarity).

---

## Training Options

```bash
./train.sh --model chatterbox --audio /path/to/audio [--batch N] [--epochs N]
```

| Option | Default | Notes |
|--------|---------|-------|
| `--batch` | 4 | 16 for 24GB GPU, 4 for 10GB |
| `--epochs` | 50 | Increase for more data |

Or use environment variables: `BATCH_SIZE=16 NUM_EPOCHS=100 ./train.sh ...`

---

## Output

After training, the finetuned model is saved to:
```
chatterbox/chatterbox_output/t3_finetuned.safetensors
```

Test it:
```bash
cd chatterbox
python inference.py
# Output: output_finetuned.wav
```

---

## Models

| Model | Status | Architecture | Languages |
|-------|--------|--------------|-----------|
| [Chatterbox](chatterbox/) | Ready | Llama T3 + S3Gen | 23 languages incl. Dutch |
| [Qwen3-TTS](qwen3-tts/) | Planned | GPT-2 / SFT | Multilingual |

---

## Upstream References
- Chatterbox finetuning: [gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)
- Qwen3-TTS finetuning: [sruckh/Qwen3-TTS-finetune](https://github.com/sruckh/Qwen3-TTS-finetune)
