import json

import numpy as np
import pandas as pd

from overfit.data_quality import (
    audit_ohlcv,
    exclude_incomplete_candles,
    write_data_quality_report,
)


DAY = 86400


def _frame(times, volumes=None):
    size = len(times)
    return pd.DataFrame(
        {
            "time": times,
            "open": np.full(size, 100.0),
            "high": np.full(size, 101.0),
            "low": np.full(size, 99.0),
            "close": np.full(size, 100.0),
            "volume": volumes if volumes is not None else np.ones(size),
        }
    )


def test_audit_accepts_valid_candles_and_daily_dst_sized_intervals():
    frame = _frame([0, DAY, 2 * DAY + 3600])

    report = audit_ohlcv(frame, "D", now=3 * DAY + 3600)

    assert report["status"] == "ok"
    assert report["critical_errors"] == []
    assert report["gaps"] == []


def test_audit_marks_duplicates_and_out_of_order_rows_critical():
    frame = _frame([0, DAY, DAY, 0])

    report = audit_ohlcv(frame, "D", now=10 * DAY)

    assert report["status"] == "critical"
    assert report["duplicate_timestamps"]
    assert report["out_of_order_rows"]


def test_audit_reports_invalid_ohlc_and_negative_volume():
    frame = _frame([0, DAY])
    frame.loc[1, ["high", "low", "volume"]] = [90, 110, -1]

    report = audit_ohlcv(frame, "D", now=3 * DAY)

    assert report["status"] == "critical"
    assert report["invalid_candles"]["count"] == 1


def test_audit_rejects_missing_or_duplicate_columns():
    missing = audit_ohlcv(pd.DataFrame({"time": []}), "D", now=0)

    frame = _frame([0])
    frame.columns = ["time", "open", "high", "low", "close", "close"]
    duplicate = audit_ohlcv(frame, "D", now=DAY)

    assert missing["status"] == "critical"
    assert duplicate["status"] == "critical"
    assert duplicate["duplicate_columns"] == ["close"]


def test_audit_rejects_unrepresentable_timestamps():
    report = audit_ohlcv(_frame([1e30]), "D", now=0)

    assert report["status"] == "critical"
    assert report["invalid_timestamp_rows"]["count"] == 1


def test_audit_reports_gaps_zero_volume_and_stale_last_candle():
    frame = _frame([0, DAY, 3 * DAY, 4 * DAY], volumes=[0, 1, 1, 1])

    report = audit_ohlcv(frame, "D", now=8 * DAY)

    assert report["status"] == "warning"
    assert len(report["gaps"]) == 1
    assert report["gaps"][0]["missing_candles_estimate"] == 1
    assert report["zero_volume"]["count"] == 1
    assert report["last_candle"]["stale"] is True


def test_incomplete_candle_is_reported_then_explicitly_excluded():
    frame = _frame([0, DAY, 2 * DAY])
    now = 2 * DAY + 100

    report = audit_ohlcv(frame, "D", now=now)
    closed, excluded = exclude_incomplete_candles(frame, "D", now=now)

    assert report["last_candle"]["incomplete"] is True
    assert excluded == 1
    assert len(closed) == len(frame) - 1
    assert len(frame) == 3


def test_quality_report_writes_json_and_gap_csv(tmp_path):
    frame = _frame([0, DAY, 3 * DAY])
    report = audit_ohlcv(frame, "D", now=5 * DAY)

    write_data_quality_report(report, tmp_path)

    saved = json.loads((tmp_path / "data_quality.json").read_text())
    gaps = pd.read_csv(tmp_path / "data_gaps.csv")
    assert saved["status"] == "warning"
    assert len(saved["gaps"]) == len(gaps) == 1
