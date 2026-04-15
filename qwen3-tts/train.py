"""
train.py — Fine-tune Qwen3-TTS on a custom voice.

Reads FileBasedDataset/train_with_codes.jsonl produced by prepare_dataset.py
and trains the T3 talker using Accelerate (bf16, gradient accumulation).

Features:
  - 90/10 train/val split with validation loss after every N epochs
  - Overfitting status display (still learning / going well / watch out / probably overfitting)
  - Automatic early stopping when val_loss stops improving
  - Checkpoints every SAVE_EVERY_EPOCHS epochs (default 1)
  - Best checkpoint kept separately in qwen3_output/best/

Checkpoints are saved as complete model directories:
  qwen3_output/checkpoint-epoch-0/
  qwen3_output/checkpoint-epoch-1/
  qwen3_output/best/   ← best val_loss so far

Usage:
    python train.py          # uses src/config.py defaults + env var overrides
    NUM_EPOCHS=20 python train.py
    SAVE_EVERY_EPOCHS=5 EARLY_STOPPING_PATIENCE=3 python train.py
"""

import json
import math
import os
import shutil
import sys

import torch

# Expose repo root so common/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Qwen3TrainConfig
from src.dataset import TTSDataset
from common.training_status import get_status


def get_attention_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def run_forward(model, batch):
    """Execute one forward pass and return (loss, sub_talker_loss)."""
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

    return outputs.loss + sub_talker_loss


def evaluate(model, val_dataloader, accelerator) -> float:
    """Run the validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in val_dataloader:
            loss = run_forward(model, batch)
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


def save_checkpoint(cfg, accelerator, model, config_dict, target_speaker_embedding, epoch):
    """Save a full model checkpoint directory. Returns the directory path."""
    ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch}")
    shutil.copytree(cfg.model_dir, ckpt_dir, dirs_exist_ok=True)

    # Write updated config with speaker metadata
    updated_cfg = dict(config_dict)
    updated_cfg["tts_model_type"] = "custom_voice"
    talker_cfg = updated_cfg.get("talker_config", {})
    talker_cfg["spk_id"] = {cfg.speaker_name: 3000}
    talker_cfg["spk_is_dialect"] = {cfg.speaker_name: False}
    updated_cfg["talker_config"] = talker_cfg

    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(updated_cfg, f, indent=2, ensure_ascii=False)

    # Save weights (drop speaker encoder, inject speaker embedding at index 3000)
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

    from safetensors.torch import save_file
    save_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
    return ckpt_dir


def main():
    cfg = Qwen3TrainConfig()

    print("==========================================")
    print("  Qwen3-TTS Finetuning")
    print(f"  Model  : {cfg.model_dir}")
    print(f"  Output : {cfg.output_dir}")
    print(f"  Epochs : {cfg.num_epochs}")
    print(f"  Batch  : {cfg.batch_size} (accum {cfg.grad_accum})")
    print(f"  Speaker: {cfg.speaker_name}")
    print(f"  Val split: {cfg.val_split:.0%} | Save every: {cfg.save_every_epochs} epoch(s) | Early stop patience: {cfg.early_stopping_patience}")
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
    with open(os.path.join(cfg.model_dir, "config.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # --- Dataset split ---
    print(f"Loading training data from {cfg.codes_path} …")
    with open(cfg.codes_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    n_val = max(1, math.floor(len(all_data) * cfg.val_split))
    n_train = len(all_data) - n_val
    train_data = all_data[:n_train]
    val_data = all_data[n_train:]
    print(f"  {len(all_data)} total samples → {n_train} train / {n_val} val\n")

    train_dataset = TTSDataset(train_data, qwen3tts.processor, config)
    val_dataset = TTSDataset(val_data, qwen3tts.processor, config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, val_dataloader
    )

    target_speaker_embedding = None
    model.train()

    # --- Training state ---
    best_val_loss = float("inf")
    prev_val_loss = None
    no_improve = 0
    best_ckpt_dir = None

    # --- Training loop ---
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = run_forward(model, batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                # Capture speaker embedding on first batch
                if target_speaker_embedding is None:
                    with torch.no_grad():
                        ref_mels = batch["ref_mels"]
                        target_speaker_embedding = model.speaker_encoder(
                            ref_mels.to(model.device).to(model.dtype)
                        ).detach()

            epoch_loss += loss.item()
            num_steps += 1

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch + 1}/{cfg.num_epochs} | Step {step} | Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(num_steps, 1)

        # --- Validation ---
        val_loss = evaluate(model, val_dataloader, accelerator)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        status = get_status(avg_loss, val_loss, best_val_loss, prev_val_loss)
        accelerator.print(
            f"[Epoch {epoch + 1}] "
            f"train={avg_loss:.4f}  "
            f"val={val_loss:.4f}  "
            f"best_val={best_val_loss:.4f}  "
            f"status='{status}'"
        )

        # --- Save checkpoint every N epochs ---
        if accelerator.is_main_process and (epoch + 1) % cfg.save_every_epochs == 0:
            ckpt_dir = save_checkpoint(cfg, accelerator, model, config_dict, target_speaker_embedding, epoch)
            accelerator.print(f"Checkpoint saved: {ckpt_dir}")

            # Keep a copy of the best checkpoint
            if val_loss <= best_val_loss:
                best_ckpt_dir = os.path.join(cfg.output_dir, "best")
                if os.path.exists(best_ckpt_dir):
                    shutil.rmtree(best_ckpt_dir)
                shutil.copytree(ckpt_dir, best_ckpt_dir)
                accelerator.print(f"Best checkpoint updated: {best_ckpt_dir}")

        # --- Early stopping ---
        if prev_val_loss is not None and val_loss >= best_val_loss:
            no_improve += 1
        else:
            no_improve = 0

        prev_val_loss = val_loss

        if no_improve >= cfg.early_stopping_patience:
            accelerator.print(
                f"\nEarly stopping triggered after {no_improve} evaluations without improvement."
            )
            if best_ckpt_dir:
                accelerator.print(f"Best model at: {best_ckpt_dir}")
            break

    print("\n==========================================")
    print("  Training complete!")
    print(f"  Checkpoints: {cfg.output_dir}")
    if best_ckpt_dir:
        print(f"  Best model : {best_ckpt_dir}")
    print("  Use the best (or last) checkpoint for inference:")
    print(f"    python inference.py")
    print("==========================================\n")


if __name__ == "__main__":
    main()
