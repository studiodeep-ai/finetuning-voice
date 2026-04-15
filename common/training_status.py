"""
training_status.py — Shared overfitting status logic for training monitors.

Used by chatterbox (via TrainingMonitorCallback) and qwen3-tts (directly in
the training loop) to produce a human-readable training health label.
"""

OVERFIT_HARD = 1.30    # val_loss > best * 1.30
OVERFIT_SOFT = 1.15    # val_loss > best * 1.15
CONVERGE_DELTA = 0.01  # val_loss change smaller than this → stable


def get_status(train_loss: float, val_loss: float, best_val: float, prev_val) -> str:
    """Return a human-readable training health label.

    Returns one of:
      'first eval'           — no previous val to compare against
      'still learning'       — val_loss still decreasing
      'going well'           — val_loss stable, close to train_loss
      'watch out'            — val_loss 15–30% above its best
      'probably overfitting' — val_loss 30%+ above its best while train < val
    """
    ratio = val_loss / best_val if best_val > 0 else 1.0

    if prev_val is None:
        return "first eval"

    val_dropped = val_loss < prev_val - CONVERGE_DELTA
    gap = val_loss - train_loss  # positive = val worse than train

    if ratio >= OVERFIT_HARD and gap > 0.05:
        return "probably overfitting"
    if ratio >= OVERFIT_SOFT and gap > 0.02:
        return "watch out"
    if val_dropped:
        return "still learning"
    return "going well"
