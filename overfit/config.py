"""Static configuration and experiment variants."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    symbol: str = "BTCUSDT"
    resolution: str = "D"  # "D" (daily) or "60" (hourly)
    start: str = "2021-01-01"
    end: str | None = None  # None means "now"
    cache_dir: Path = Path("data")
    report_dir: Path = Path("reports")
    request_pause: float = 3.1  # stay under 20 req/min: the docs list 20 and 60 ambiguously
    train_size: int = 365
    test_size: int = 60
    val_fraction: float = 0.2  # chronological tail of the train window
    epochs: int = 200
    quick_epochs: int = 20
    threads: int = 2
    lr: float = 1e-2
    seed: int = 42


@dataclass(frozen=True)
class Variant:
    """One model/regularization setup. hidden=0 means logistic regression."""

    hidden: int
    dropout: float = 0.0
    weight_decay: float = 0.0
    patience: int | None = None  # None disables early stopping


VARIANTS: dict[str, Variant] = {
    "logistic": Variant(hidden=0),
    "mlp": Variant(hidden=64),
    "dropout": Variant(hidden=64, dropout=0.3),
    "weight_decay": Variant(hidden=64, weight_decay=1.0),
    "early_stop": Variant(hidden=64, patience=15),
    "all": Variant(hidden=64, dropout=0.3, weight_decay=1.0, patience=15),
}
