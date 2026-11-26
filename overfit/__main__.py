"""CLI entry point: python -m overfit [--symbol BTCUSDT] [--refresh]."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

import requests

from .config import Config
from .data import DataError, build_dataset, load_ohlcv
from .report import plot_summary, summarize
from .validate import run_multiple_seeds

log = logging.getLogger("overfit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=Config.symbol)
    parser.add_argument("--resolution", default=Config.resolution, choices=["D", "60"])
    parser.add_argument("--refresh", action="store_true", help="re-download candles")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        metavar="SEED",
        help="random seeds to evaluate (default: 42 43 44 45 46)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = replace(Config(), symbol=args.symbol, resolution=args.resolution)

    try:
        x, y = build_dataset(load_ohlcv(cfg, refresh=args.refresh))
        results = run_multiple_seeds(x, y, cfg, tuple(args.seeds))
    except (DataError, ValueError, requests.RequestException) as exc:
        log.error("%s", exc)
        return 1

    summary = summarize(results)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.report_dir / "folds.csv", index=False)
    summary.to_csv(cfg.report_dir / "summary.csv")
    plot_summary(summary, cfg.report_dir / "summary.png")
    print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
