"""Small PyTorch models and a full-batch trainer with optional early stopping."""
from __future__ import annotations

import torch
from torch import nn

from .config import Config, Variant


def build_model(n_features: int, variant: Variant) -> nn.Module:
    """Logistic regression (hidden=0) or a two-layer MLP, a few thousand params."""
    if variant.hidden == 0:
        return nn.Linear(n_features, 1)
    return nn.Sequential(
        nn.Linear(n_features, variant.hidden),
        nn.ReLU(),
        nn.Dropout(variant.dropout),
        nn.Linear(variant.hidden, variant.hidden),
        nn.ReLU(),
        nn.Dropout(variant.dropout),
        nn.Linear(variant.hidden, 1),
    )


def fit(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    variant: Variant,
    cfg: Config,
) -> nn.Module:
    """Train on (x_fit, y_fit); x_val only drives early stopping when enabled.

    The data is tiny, so full-batch steps are both the simplest and fastest.
    """
    torch.manual_seed(cfg.seed)
    model = build_model(x_fit.shape[1], variant)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=variant.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_state, stale = float("inf"), None, 0

    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(x_fit).squeeze(-1), y_fit).backward()
        optimizer.step()

        if variant.patience is None:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val).squeeze(-1), y_val).item()
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= variant.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Return (loss, accuracy); the model must already be in eval mode."""
    logits = model(x).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
    accuracy = ((logits > 0).float() == y).float().mean().item()
    return loss, accuracy
