import logging
import os
import sys
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback
import torch as _torch


class ChatterboxTrainer(Trainer):
    """Trainer subclass that always computes loss during evaluation.

    The HF Trainer's prediction_step skips loss computation when the batch
    contains no 'labels' key. Our data collator doesn't include one, so we
    override prediction_step to always run the model and return the loss.
    """
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with _torch.no_grad():
            loss, _ = model(**inputs)
        return (loss.detach(), None, None)
from safetensors.torch import save_file

from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator_turbo, data_collator_standart
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.preprocess_file_based import preprocess_dataset_file_based
from src.preprocess_json import preprocess_dataset_json_based
from src.utils import setup_logger, check_pretrained_models

from src.inference_callback import InferenceCallback
from src.training_monitor import TrainingMonitorCallback

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3
from src.training_ui import TrainingDashboard

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def main():

    cfg = TrainConfig()

    # --- Redirect all logging to a file so the Rich dashboard stays clean ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = os.path.join(cfg.output_dir, "training.log")
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
    root.setLevel(logging.INFO)
    # Suppress transformers' own verbose output
    import transformers as _tf
    _tf.logging.set_verbosity_error()

    logger.info("--- Starting Chatterbox Finetuning ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 1. SELECT THE CORRECT ENGINE CLASS
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS
    
    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    # Loading on CPU first to save VRAM
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    # We prevent caching during training.
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)

    # 4. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)


    # --- SPECIAL SETTING FOR TURBO ---
    if cfg.is_turbo:
        logger.info("Turbo Mode: Removing backbone WTE layer...")
        if hasattr(new_t3_model.tfmr, "wte"):
            del new_t3_model.tfmr.wte


    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 5. PREPARE ENGINE FOR TRAINING
    # Reload engine components (VoiceEncoder, S3Gen) but inject our new T3
    tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model 

    # Freeze other components
    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters(): 
        param.requires_grad = False
        
    for param in tts_engine_new.s3gen.parameters(): 
        param.requires_grad = False

    # Enable Training for T3
    tts_engine_new.t3.train()
    for param in tts_engine_new.t3.parameters(): 
        param.requires_grad = True

    if cfg.preprocess:
        
        logger.info("Initializing Preprocess dataset...")
        
        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, tts_engine_new)
            
        elif cfg.json_format:
            preprocess_dataset_json_based(cfg, tts_engine_new)
            
        else:
            preprocess_dataset_file_based(cfg, tts_engine_new)
      
    else:
        logger.info("Skipping the preprocessing dataset step...")
            
        
    # 6. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    full_ds = ChatterboxDataset(cfg)

    # Train / validation split
    import math as _math
    from torch.utils.data import Subset
    n_total = len(full_ds)
    n_val = max(1, _math.floor(n_total * cfg.val_split))
    n_train = n_total - n_val
    indices = list(range(n_total))
    train_ds = Subset(full_ds, indices[:n_train])
    val_ds   = Subset(full_ds, indices[n_train:])
    logger.info(f"Dataset split: {n_train} train / {n_val} val (val_split={cfg.val_split})")

    model_label = "Chatterbox-Turbo" if cfg.is_turbo else "Chatterbox"
    dashboard = TrainingDashboard(model_name=model_label, total_epochs=cfg.num_epochs)

    trainer_callbacks = [
        TrainingMonitorCallback(dashboard=dashboard),
        EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
    ]
    if cfg.is_inference:
        inference_cb = InferenceCallback(cfg)
        trainer_callbacks.append(inference_cb)

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    if cfg.is_turbo:
        logger.info("Using Turbo Data Collator (with dynamic prompt masking)")
        selected_collator = data_collator_turbo
    else:
        logger.info("Using Standard Data Collator")
        selected_collator = data_collator_standart

    # Compute save_steps so checkpoints are written every N epochs.
    import math
    steps_per_epoch = math.ceil(n_train / (cfg.batch_size * cfg.grad_accum))
    save_steps = max(1, steps_per_epoch * cfg.save_every_epochs)
    logger.info(f"Steps per epoch: {steps_per_epoch} | Saving/evaluating every {cfg.save_every_epochs} epochs ({save_steps} steps)")

    # 7. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=save_steps,       # Evaluate at the same frequency as saving
        logging_strategy="epoch",
        remove_unused_columns=False, # Required for our custom wrapper
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=False,
        bf16=device.type == "cuda",  # bf16 only works on CUDA; MPS/CPU use fp32
        save_total_limit=cfg.save_total_limit,
        save_only_model=True,          # Skip optimizer/scheduler state — saves GBs of disk space
        load_best_model_at_end=True,   # Restore best checkpoint when training ends/stops early
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=device.type == "cuda",  # gradient checkpointing is CUDA-only
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
    )

    trainer = ChatterboxTrainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=selected_collator,
        callbacks=trainer_callbacks,
    )

    # Remove HF's default noisy callbacks — our dashboard replaces them
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)

    logger.info("Starting Training Loop...")
    trainer.train()


    # 8. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    filename = "t3_turbo_finetuned.safetensors" if cfg.is_turbo else "t3_finetuned.safetensors"
    final_model_path = os.path.join(cfg.output_dir, filename)

    save_file(tts_engine_new.t3.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__": 
    main()
