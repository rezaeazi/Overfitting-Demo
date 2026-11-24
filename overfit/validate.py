"""Walk-forward validation: compare in-sample and out-of-sample performance."""
from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch

from .config import VARIANTS, Config
from .model import evaluate, fit

log = logging.getLogger(__name__)


def walk_forward(n: int, train_size: int, test_size: int) -> Iterator[tuple[slice, slice]]:
    """Rolling windows: train on `train_size` rows, test on the next `test_size`.

    Labels look one step ahead, so the test window can start right after the
    train window without leaking future information.
    """
    for start in range(0, n - train_size - test_size + 1, test_size):
        train_end = start + train_size
        yield slice(start, train_end), slice(train_end, train_end + test_size)


CLIP = 5.0  # winsorise z-scores so regime shifts cannot blow up the logits


def _tensor(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.clip((a - mean) / std, -CLIP, CLIP))


def run(x: np.ndarray, y: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Return one row per (fold, variant) with train/test loss and accuracy."""
    if len(x) < cfg.train_size + cfg.test_size:
        raise ValueError(
            f"need at least {cfg.train_size + cfg.test_size} rows, got {len(x)}"
        )
    rows = []
    for fold, (train, test) in enumerate(walk_forward(len(x), cfg.train_size, cfg.test_size)):
        n_fit = int(cfg.train_size * (1 - cfg.val_fraction))
        x_train, y_train = x[train], y[train]
        x_fit, y_fit = x_train[:n_fit], y_train[:n_fit]
        y_test = y[test]

        # Scaling statistics come from the fit rows only (no leakage).
        mean, std = x_fit.mean(axis=0), x_fit.std(axis=0) + 1e-8
        t_fit, t_val = _tensor(x_fit, mean, std), _tensor(x_train[n_fit:], mean, std)
        t_test = _tensor(x[test], mean, std)
        ty_fit, ty_val, ty_test = (torch.from_numpy(a) for a in (y_fit, y_train[n_fit:], y_test))

        majority = float(y_fit.mean() >= 0.5)
        baseline = float((y_test == majority).mean())

        for name, variant in VARIANTS.items():
            model = fit(t_fit, ty_fit, t_val, ty_val, variant, cfg)
            train_loss, train_acc = evaluate(model, t_fit, ty_fit)
            test_loss, test_acc = evaluate(model, t_test, ty_test)
            rows.append(
                {
                    "fold": fold,
                    "variant": name,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "baseline_acc": baseline,
                }
            )
        log.info("fold %d done", fold)
    return pd.DataFrame(rows)
