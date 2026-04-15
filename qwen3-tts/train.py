"""
train.py — Fine-tune Qwen3-TTS on a custom voice.

Reads FileBasedDataset/train_with_codes.jsonl produced by prepare_dataset.py
and trains the T3 talker using Accelerate (bf16, gradient accumulation).

Checkpoints are saved as complete model directories to qwen3_output/:
  qwen3_output/checkpoint-epoch-0/
  qwen3_output/checkpoint-epoch-1/
  ...

Each checkpoint can be used directly for inference.

Usage:
    python train.py          # uses src/config.py defaults + env var overrides
    NUM_EPOCHS=20 python train.py
"""

import json
import os
import shutil
import sys

import torch

# Add parent src/ to path so imports work when called from the model dir
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Qwen3TrainConfig
from src.dataset import TTSDataset


def get_attention_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def main():
    cfg = Qwen3TrainConfig()

    print("==========================================")
    print("  Qwen3-TTS Finetuning")
    print(f"  Model  : {cfg.model_dir}")
    print(f"  Output : {cfg.output_dir}")
    print(f"  Epochs : {cfg.num_epochs}")
    print(f"  Batch  : {cfg.batch_size} (accum {cfg.grad_accum})")
    print(f"  Speaker: {cfg.speaker_name}")
    print("==========================================\n")

    # --- Validate inputs ---
    if not os.path.isfile(cfg.codes_path):
        print(f"ERROR: Training data not found: {cfg.codes_path}")
        print("Run prepare_dataset.py first.")
        sys.exit(1)

    if not os.path.isdir(cfg.model_dir):
        print(f"ERROR: Pretrained model not found: {cfg.model_dir}")
        print("Run setup.py first.")
        sys.exit(1)

    # --- Imports ---
    from accelerate import Accelerator
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    attn_impl = get_attention_implementation()
    print(f"Attention implementation: {attn_impl}\n")

    # --- Accelerator ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=log_dir,
    )

    # --- Model ---
    print("Loading pretrained model …")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        cfg.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    config = AutoConfig.from_pretrained(cfg.model_dir)

    # --- Dataset ---
    print(f"Loading training data from {cfg.codes_path} …")
    with open(cfg.codes_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    print(f"  {len(train_data)} training samples\n")

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    model, optimizer, dataloader = accelerator.prepare(qwen3tts.model, optimizer, dataloader)

    target_speaker_embedding = None
    model.train()

    # --- Training loop ---
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()

                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_emb = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    input_embeddings = input_embeddings + codec_i_emb * codec_mask.unsqueeze(-1)

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_steps += 1

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch + 1}/{cfg.num_epochs} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(num_steps, 1)
        accelerator.print(f"Epoch {epoch + 1} complete — avg loss: {avg_loss:.4f}")

        # --- Save checkpoint ---
        if accelerator.is_main_process:
            ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch}")
            shutil.copytree(cfg.model_dir, ckpt_dir, dirs_exist_ok=True)

            # Update config with speaker info
            config_path = os.path.join(ckpt_dir, "config.json")
            with open(os.path.join(cfg.model_dir, "config.json"), "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"
            talker_cfg = config_dict.get("talker_config", {})
            talker_cfg["spk_id"] = {cfg.speaker_name: 3000}
            talker_cfg["spk_is_dialect"] = {cfg.speaker_name: False}
            config_dict["talker_config"] = talker_cfg

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Save weights (drop speaker encoder, inject speaker embedding)
            unwrapped = accelerator.unwrap_model(model)
            state_dict = {
                k: v.detach().cpu().to(torch.float32)
                for k, v in unwrapped.state_dict().items()
                if not k.startswith("speaker_encoder")
            }
            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )

            save_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
            print(f"Checkpoint saved: {ckpt_dir}")

    print("\n==========================================")
    print("  Training complete!")
    print(f"  Checkpoints: {cfg.output_dir}")
    print("  Use the last checkpoint for inference:")
    print(f"    python inference.py")
    print("==========================================\n")


if __name__ == "__main__":
    main()
