"""
training_monitor.py — Callback that prints validation loss + overfitting status
after each evaluation step.

Status logic:
  - "still learning"       — val_loss still decreasing (or first eval)
  - "going well"           — val_loss stable, close to train_loss
  - "watch out"            — val_loss 15–30% above its best, train_loss keeps dropping
  - "probably overfitting" — val_loss 30%+ above its best while train_loss is lower
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # expose common/

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from common.training_status import get_status
from src.utils import setup_logger

logger = setup_logger("TrainingMonitor")


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.best_val_loss = float("inf")
        self.prev_val_loss = None
        self.last_train_loss = None

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if not metrics:
            return

        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        train_loss = self.last_train_loss or val_loss
        status = get_status(train_loss, val_loss, self.best_val_loss, self.prev_val_loss)

        epoch = metrics.get("epoch", state.epoch or 0)

        logger.info(
            f"[Epoch {epoch:.0f}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"best_val={self.best_val_loss:.4f}  "
            f"status='{status}'"
        )

        self.prev_val_loss = val_loss
