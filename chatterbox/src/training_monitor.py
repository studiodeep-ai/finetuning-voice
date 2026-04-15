"""
training_monitor.py — HF Trainer callbacks that drive the Rich training dashboard.

When a TrainingDashboard is provided:
  - on_train_begin  → starts the live display
  - on_step_end     → updates the step progress bar (throttled)
  - on_log          → captures the latest train loss
  - on_evaluate     → updates stats panel + checkpoint history
  - on_train_end    → stops the display, prints a plain summary

Without a dashboard (fallback mode), status lines are written to the logger.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # expose common/

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from common.training_status import get_status
from src.utils import setup_logger

logger = setup_logger("TrainingMonitor")


class TrainingMonitorCallback(TrainerCallback):
    """Drives the Rich dashboard (or falls back to plain logging if no dashboard)."""

    def __init__(self, dashboard=None):
        from src.training_ui import TrainingDashboard
        self.dashboard: Optional[TrainingDashboard] = dashboard

        self.best_val_loss: float = float("inf")
        self.prev_val_loss: Optional[float] = None
        self.last_train_loss: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.dashboard:
            self.dashboard.start()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.dashboard:
            self.dashboard.stop()
            self.dashboard.print_summary()

    # ------------------------------------------------------------------
    # Per-step: update progress bar
    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.dashboard and state.max_steps:
            self.dashboard.update_step(
                global_step=state.global_step,
                max_steps=state.max_steps,
                epoch=state.epoch or 0,
            )

    # ------------------------------------------------------------------
    # Per-log: capture train loss
    # ------------------------------------------------------------------

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    # ------------------------------------------------------------------
    # Per-eval: update stats + checkpoint history
    # ------------------------------------------------------------------

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        if not metrics:
            return

        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        train_loss = self.last_train_loss or val_loss
        status = get_status(train_loss, val_loss, self.best_val_loss, self.prev_val_loss)
        epoch = int(metrics.get("epoch", state.epoch or 0))

        if self.dashboard:
            self.dashboard.update_stats(train_loss, val_loss, self.best_val_loss, status)
            self.dashboard.add_checkpoint(epoch, train_loss, val_loss, status)
        else:
            # Fallback: plain log line when running without a terminal
            logger.info(
                f"[Epoch {epoch}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"best_val={self.best_val_loss:.4f}  "
                f"status='{status}'"
            )

        self.prev_val_loss = val_loss
