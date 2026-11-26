"""OHLCV quality checks and explicit handling of in-progress candles."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, STEP_SECONDS

SAMPLE_LIMIT = 20


class DataQualityError(RuntimeError):
    """Raised when OHLCV data has critical integrity problems."""


def _utc_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def audit_ohlcv(
    frame: pd.DataFrame, resolution: str, now: float | None = None
) -> dict[str, object]:
    """Inspect OHLCV rows without sorting, dropping, or filling any data."""
    if resolution not in STEP_SECONDS:
        raise ValueError(f"resolution must be one of {sorted(STEP_SECONDS)}")

    step = STEP_SECONDS[resolution]
    checked_at = time.time() if now is None else float(now)
    critical_errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "status": "ok",
        "checked_at_utc": datetime.fromtimestamp(
            checked_at, tz=timezone.utc
        ).isoformat(),
        "resolution": resolution,
        "row_count": int(len(frame)),
        "critical_errors": critical_errors,
        "warnings": warnings,
        "duplicate_columns": [],
        "duplicate_timestamps": [],
        "out_of_order_rows": [],
        "invalid_candles": {"count": 0, "examples": []},
        "gaps": [],
        "zero_volume": {"count": 0, "examples": []},
        "last_candle": None,
        "excluded_incomplete_candles": 0,
    }

    missing_columns = sorted(set(COLUMNS) - set(frame.columns))
    duplicate_columns = [
        str(column) for column in frame.columns[frame.columns.duplicated()].unique()
    ]
    if missing_columns:
        critical_errors.append(f"missing required columns: {missing_columns}")
    if duplicate_columns:
        report["duplicate_columns"] = duplicate_columns
        critical_errors.append(f"duplicate columns: {duplicate_columns}")
    if frame.empty:
        critical_errors.append("no candles to audit")
    if missing_columns or duplicate_columns or frame.empty:
        report["status"] = "critical"
        return report

    numeric = frame[COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    timestamps = values[:, 0]
    valid_timestamps = (
        np.isfinite(timestamps)
        & (timestamps == np.floor(timestamps))
        & (timestamps >= 0)
        & (timestamps <= 253402300799)
    )
    invalid_timestamp_rows = np.flatnonzero(~valid_timestamps)
    if len(invalid_timestamp_rows):
        examples = [int(index) for index in invalid_timestamp_rows[:SAMPLE_LIMIT]]
        report["invalid_timestamp_rows"] = {
            "count": int(len(invalid_timestamp_rows)),
            "examples": examples,
        }
        critical_errors.append(
            f"{len(invalid_timestamp_rows)} rows have invalid timestamps"
        )

    invalid_candles: list[dict[str, object]] = []
    for index, row in enumerate(values):
        candle_time, open_price, high, low, close, volume = row
        reasons = []
        if not np.isfinite(row).all():
            reasons.append("non-finite or non-numeric OHLCV value")
        else:
            if min(open_price, high, low, close) <= 0:
                reasons.append("OHLC prices must be positive")
            if high < max(open_price, close):
                reasons.append("high is below open or close")
            if low > min(open_price, close):
                reasons.append("low is above open or close")
            if high < low:
                reasons.append("high is below low")
            if volume < 0:
                reasons.append("volume is negative")
        if reasons:
            item: dict[str, object] = {"row": int(index), "reasons": reasons}
            if valid_timestamps[index]:
                item["time"] = int(candle_time)
            invalid_candles.append(item)
    if invalid_candles:
        report["invalid_candles"] = {
            "count": len(invalid_candles),
            "examples": invalid_candles[:SAMPLE_LIMIT],
        }
        critical_errors.append(f"{len(invalid_candles)} candles have invalid values")

    zero_volume_rows = np.flatnonzero(
        np.isfinite(values[:, 5]) & (values[:, 5] == 0)
    )
    report["zero_volume"] = {
        "count": int(len(zero_volume_rows)),
        "examples": [
            {
                "row": int(index),
                "time": int(timestamps[index]) if valid_timestamps[index] else None,
            }
            for index in zero_volume_rows[:SAMPLE_LIMIT]
        ],
    }
    if len(zero_volume_rows):
        warnings.append(f"{len(zero_volume_rows)} candles have zero volume")

    valid_times = timestamps[valid_timestamps].astype(np.int64)
    unique_times, counts = np.unique(valid_times, return_counts=True)
    duplicates = [
        {"time": int(candle_time), "count": int(count)}
        for candle_time, count in zip(unique_times, counts)
        if count > 1
    ]
    if duplicates:
        report["duplicate_timestamps"] = duplicates[:SAMPLE_LIMIT]
        duplicate_rows = sum(item["count"] - 1 for item in duplicates)
        critical_errors.append(f"{duplicate_rows} duplicate timestamp rows")

    out_of_order = [
        {
            "row": int(index),
            "previous_time": int(timestamps[index - 1]),
            "time": int(timestamps[index]),
        }
        for index in range(1, len(timestamps))
        if valid_timestamps[index - 1]
        and valid_timestamps[index]
        and timestamps[index] < timestamps[index - 1]
    ]
    if out_of_order:
        report["out_of_order_rows"] = out_of_order[:SAMPLE_LIMIT]
        critical_errors.append(f"{len(out_of_order)} out-of-order candle rows")

    future_rows = np.flatnonzero(valid_timestamps & (timestamps > checked_at))
    if len(future_rows):
        report["future_timestamp_rows"] = {
            "count": int(len(future_rows)),
            "examples": [int(index) for index in future_rows[:SAMPLE_LIMIT]],
        }
        critical_errors.append(
            f"{len(future_rows)} candle timestamps are in the future"
        )

    gaps = []
    for index in range(1, len(timestamps)):
        if not (valid_timestamps[index - 1] and valid_timestamps[index]):
            continue
        delta = int(timestamps[index] - timestamps[index - 1])
        gap_tolerance = 3600 if resolution == "D" else step * 0.5
        if delta > step + gap_tolerance:
            gaps.append(
                {
                    "from_utc": _utc_timestamp(int(timestamps[index - 1])),
                    "to_utc": _utc_timestamp(int(timestamps[index])),
                    "elapsed_seconds": delta,
                    "missing_candles_estimate": max(
                        int(round(delta / step)) - 1, 1
                    ),
                }
            )
    report["gaps"] = gaps
    if gaps:
        warnings.append(f"{len(gaps)} gaps exceed the expected candle interval")

    if valid_timestamps[-1]:
        last_time = int(timestamps[-1])
        close_time = last_time + step
        incomplete = last_time <= checked_at < close_time
        stale = checked_at - close_time > 2 * step
        report["last_candle"] = {
            "time_utc": _utc_timestamp(last_time),
            "expected_close_utc": _utc_timestamp(close_time),
            "incomplete": bool(incomplete),
            "stale": bool(stale),
        }
        if incomplete:
            warnings.append("last candle is still in progress")
        if stale:
            warnings.append("last candle is stale by more than two intervals")

    if critical_errors:
        report["status"] = "critical"
    elif warnings:
        report["status"] = "warning"
    return report


def exclude_incomplete_candles(
    frame: pd.DataFrame, resolution: str, now: float | None = None
) -> tuple[pd.DataFrame, int]:
    """Explicitly remove candles that have not closed yet; never fills gaps."""
    if resolution not in STEP_SECONDS:
        raise ValueError(f"resolution must be one of {sorted(STEP_SECONDS)}")
    if "time" not in frame.columns:
        raise DataQualityError("cannot exclude incomplete candles without timestamps")

    checked_at = time.time() if now is None else float(now)
    timestamps = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(timestamps).all():
        raise DataQualityError("cannot filter candles with invalid timestamps")
    if (timestamps > checked_at).any():
        raise DataQualityError("cannot filter candles with future timestamps")
    closed = timestamps + STEP_SECONDS[resolution] <= checked_at
    excluded = int((~closed).sum())
    return frame.loc[closed].copy(), excluded


def write_data_quality_report(
    report: dict[str, object], report_dir: Path
) -> None:
    """Write the full audit and a convenient CSV of detected gaps."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "data_quality.json"
    json_temp = json_path.with_suffix(".json.tmp")
    with json_temp.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    json_temp.replace(json_path)

    gap_columns = [
        "from_utc",
        "to_utc",
        "elapsed_seconds",
        "missing_candles_estimate",
    ]
    gaps = pd.DataFrame(report["gaps"], columns=gap_columns)
    csv_path = report_dir / "data_gaps.csv"
    csv_temp = csv_path.with_suffix(".csv.tmp")
    gaps.to_csv(csv_temp, index=False)
    csv_temp.replace(csv_path)
