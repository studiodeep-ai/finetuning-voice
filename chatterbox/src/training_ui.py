"""
training_ui.py — Rich terminal dashboard for Chatterbox finetuning.

Renders a single live screen (alternate buffer) that updates in-place:
  - Step-level progress bar
  - Current train / val / best loss + overfitting status
  - Checkpoint history table (last 10 evals)

All Python logging is redirected to a file so the dashboard stays clean.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Status → display style
STATUS_STYLE = {
    "—":                    "dim",
    "first eval":           "cyan",
    "still learning":       "green",
    "going well":           "bold green",
    "watch out":            "bold yellow",
    "probably overfitting": "bold red",
}

STATUS_ICON = {
    "—":                    "·",
    "first eval":           "?",
    "still learning":       "↓",
    "going well":           "✓",
    "watch out":            "⚠",
    "probably overfitting": "✗",
}


class TrainingDashboard:
    """A Rich live terminal dashboard for monitoring training."""

    # Minimum seconds between forced re-renders (step updates are frequent)
    _REFRESH_THROTTLE = 2.0

    def __init__(self, model_name: str = "Chatterbox", total_epochs: int = 50):
        self.model_name = model_name
        self.total_epochs = total_epochs

        # --- State ---
        self.current_epoch: float = 0.0
        self.current_step: int = 0
        self.max_steps: int = 1

        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.best_val: Optional[float] = None
        self.status: str = "—"
        self.start_time: datetime = datetime.now()
        self.checkpoints: list = []

        self._last_refresh: float = 0.0

        # --- Rich progress bar (stateful — tasks are updated, not recreated) ---
        self.progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA"),
            TimeRemainingColumn(),
            expand=True,
        )
        self._step_task = self.progress.add_task("Training steps", total=1)

        # --- Live display on alternate screen (single clean frame, no scroll) ---
        self.live = Live(
            self._render(),
            screen=True,
            refresh_per_second=2,
            console=Console(),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.start_time = datetime.now()
        self.live.start()

    def stop(self):
        self.live.stop()

    # ------------------------------------------------------------------
    # Update API (called from TrainingMonitorCallback)
    # ------------------------------------------------------------------

    def update_step(self, global_step: int, max_steps: int, epoch: float):
        """Called on every training step — only re-renders at throttled rate."""
        self.current_step = global_step
        self.max_steps = max_steps
        self.current_epoch = epoch
        self.progress.update(self._step_task, total=max_steps, completed=global_step)

        now = time.time()
        if now - self._last_refresh >= self._REFRESH_THROTTLE:
            self.live.update(self._render())
            self._last_refresh = now

    def update_stats(
        self,
        train_loss: float,
        val_loss: float,
        best_val: float,
        status: str,
    ):
        """Called after each evaluation — always triggers a re-render."""
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.best_val = best_val
        self.status = status
        self.live.update(self._render())
        self._last_refresh = time.time()

    def add_checkpoint(self, epoch: int, train_loss: float, val_loss: float, status: str):
        self.checkpoints.append(
            {"epoch": epoch, "train": train_loss, "val": val_loss, "status": status}
        )
        self.live.update(self._render())
        self._last_refresh = time.time()

    def print_summary(self):
        """Print a plain-text summary after the live display exits."""
        c = Console()
        c.rule("[bold green] Training Complete ")
        if self.best_val is not None:
            c.print(f"  Best val loss : [bold green]{self.best_val:.4f}[/]")
        c.print(f"  Epochs run    : [bold]{int(self.current_epoch)}[/] / {self.total_epochs}")
        c.print(f"  Checkpoints   : [bold]{len(self.checkpoints)}[/] evals recorded")
        c.print(f"  Duration      : [bold]{self._elapsed()}[/]")
        c.rule()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _elapsed(self) -> str:
        delta = datetime.now() - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

    def _fmt(self, v) -> str:
        return f"{v:.4f}" if v is not None else "—"

    def _render(self):
        # ── Stats panel ────────────────────────────────────────────────
        st = STATUS_STYLE.get(self.status, "white")
        ic = STATUS_ICON.get(self.status, "·")

        stats_grid = Table.grid(padding=(0, 3))
        stats_grid.add_column(style="dim", min_width=14)
        stats_grid.add_column()
        stats_grid.add_row("Train loss", f"[cyan]{self._fmt(self.train_loss)}[/]")
        stats_grid.add_row("Val loss",   f"[magenta]{self._fmt(self.val_loss)}[/]")
        stats_grid.add_row("Best val",   f"[green]{self._fmt(self.best_val)}[/]")
        stats_grid.add_row("Status",     Text(f"{ic}  {self.status}", style=st))
        stats_grid.add_row("Elapsed",    self._elapsed())

        stats_panel = Panel(
            stats_grid,
            title="[bold]Latest",
            border_style="blue",
            padding=(1, 3),
        )

        # ── Checkpoint history table ────────────────────────────────────
        hist = Table(
            "Epoch", "Train loss", "Val loss", "Status",
            box=None,
            header_style="bold dim",
            show_edge=False,
            padding=(0, 3),
        )
        rows = self.checkpoints[-12:]  # last 12 evals
        if rows:
            for ckpt in rows:
                s = ckpt["status"]
                hist.add_row(
                    str(ckpt["epoch"]),
                    f"{ckpt['train']:.4f}",
                    f"{ckpt['val']:.4f}",
                    Text(f"{STATUS_ICON.get(s, '·')}  {s}", style=STATUS_STYLE.get(s, "white")),
                )
        else:
            hist.add_row("—", "—", "—", "[dim]waiting for first evaluation…[/]")

        hist_panel = Panel(
            hist,
            title="[bold]Checkpoint History",
            border_style="dim",
            padding=(1, 3),
        )

        # ── Outer frame ────────────────────────────────────────────────
        epoch_label = f"Epoch {int(self.current_epoch)} / {self.total_epochs}"

        return Panel(
            Group(self.progress, "", stats_panel, "", hist_panel),
            title=f"[bold white] {self.model_name} Finetuning [/]",
            subtitle=f"[dim] {epoch_label} [/]",
            border_style="bright_blue",
            padding=(1, 2),
        )
