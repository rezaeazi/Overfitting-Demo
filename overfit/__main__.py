"""CLI entry point: python -m overfit [--symbol BTCUSDT] [--refresh]."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter

from .config import Config
from .runtime import RunAlreadyActiveError, exclusive_run_lock, write_run_metadata

log = logging.getLogger("overfit")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolve_run_options(
    cfg: Config, quick: bool, seeds: list[int] | None
) -> tuple[Config, tuple[int, ...]]:
    if quick:
        selected_seeds = tuple(seeds) if seeds is not None else (42,)
        if len(selected_seeds) != 1:
            raise ValueError("--quick accepts exactly one seed")
        cfg = replace(cfg, epochs=min(cfg.epochs, cfg.quick_epochs))
        return cfg, selected_seeds
    return cfg, tuple(seeds) if seeds is not None else DEFAULT_SEEDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=Config.symbol)
    parser.add_argument("--resolution", default=Config.resolution, choices=["D", "60"])
    parser.add_argument("--refresh", action="store_true", help="re-download candles")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one seed with fewer epochs for a fast smoke test",
    )
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=Config.threads,
        metavar="N",
        help=f"maximum CPU threads for numerical libraries (default: {Config.threads})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="random seeds to evaluate (default: 42 43 44 45 46; --quick uses 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = replace(
        Config(),
        symbol=args.symbol,
        resolution=args.resolution,
        threads=args.threads,
    )
    try:
        cfg, seeds = _resolve_run_options(cfg, args.quick, args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(cfg.threads)

    import requests
    import torch

    from .data import DataError, build_dataset, load_ohlcv
    from .data_quality import (
        DataQualityError,
        audit_ohlcv,
        exclude_incomplete_candles,
        write_data_quality_report,
    )
    from .report import plot_summary, summarize
    from .validate import run_multiple_seeds

    try:
        with exclusive_run_lock(cfg.report_dir / ".run.lock"):
            started_at = datetime.now(timezone.utc)
            started = perf_counter()
            frame = load_ohlcv(cfg, refresh=args.refresh)
            checked_at = datetime.now(timezone.utc).timestamp()
            quality = audit_ohlcv(frame, cfg.resolution, now=checked_at)
            if quality["critical_errors"]:
                write_data_quality_report(quality, cfg.report_dir)
                raise DataQualityError(
                    "; ".join(quality["critical_errors"])
                )
            frame, excluded = exclude_incomplete_candles(
                frame, cfg.resolution, now=checked_at
            )
            quality["excluded_incomplete_candles"] = excluded
            write_data_quality_report(quality, cfg.report_dir)
            if excluded:
                log.warning(
                    "excluded %d incomplete candle(s); see reports/data_quality.json",
                    excluded,
                )
            for warning in quality["warnings"]:
                log.warning("data quality: %s", warning)
            log.info(
                "data audit: %s (%d rows, %d gaps, %d zero-volume candles)",
                quality["status"],
                quality["row_count"],
                len(quality["gaps"]),
                quality["zero_volume"]["count"],
            )
            x, y = build_dataset(frame)
            results = run_multiple_seeds(x, y, cfg, seeds)
            summary = summarize(results)
            results.to_csv(cfg.report_dir / "folds.csv", index=False)
            summary.to_csv(cfg.report_dir / "summary.csv")
            plot_summary(summary, cfg.report_dir / "summary.png")
            elapsed = perf_counter() - started
            write_run_metadata(
                cfg.report_dir / "run_metadata.json",
                {
                    "started_at": started_at.isoformat(),
                    "elapsed_seconds": round(elapsed, 3),
                    "quick": args.quick,
                    "threads": cfg.threads,
                    "torch_threads": torch.get_num_threads(),
                    "max_epochs": cfg.epochs,
                    "seeds": list(seeds),
                },
            )
            log.info("experiment completed in %.2f seconds", elapsed)
            print(summary.round(3).to_string())
    except (
        DataError,
        DataQualityError,
        ValueError,
        requests.RequestException,
        RunAlreadyActiveError,
    ) as exc:
        log.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
