"""Metrics and diagnostic plots."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             r2_score, root_mean_squared_error)

log = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_plots(y_true, y_pred, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping plots")
        return
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_true, y_pred, s=6, alpha=.4)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual", ylabel="Predicted", title="Actual vs predicted")
    ax[1].scatter(y_pred, y_true - y_pred, s=6, alpha=.4)
    ax[1].axhline(0, color="r", ls="--"); ax[1].set(xlabel="Predicted", ylabel="Residual", title="Residuals")
    fig.tight_layout(); fig.savefig(out_dir / "diagnostics.png", dpi=120); plt.close(fig)
