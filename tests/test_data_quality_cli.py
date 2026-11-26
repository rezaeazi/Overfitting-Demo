import json
import sys

import pandas as pd

from overfit import __main__ as cli
from overfit import data, validate


def test_cli_saves_critical_audit_and_stops_before_model(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        {
            "time": [0, 86400, 86400],
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["overfit", "--quick"])
    monkeypatch.setattr(data, "load_ohlcv", lambda *args, **kwargs: frame)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("critical audit must stop the run")

    monkeypatch.setattr(data, "build_dataset", fail_if_called)
    monkeypatch.setattr(validate, "run_multiple_seeds", fail_if_called)

    assert cli.main() == 1
    report = json.loads(
        (tmp_path / "reports" / "data_quality.json").read_text()
    )
    assert report["status"] == "critical"
    assert report["duplicate_timestamps"]
