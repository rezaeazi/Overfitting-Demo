import numpy as np
import pandas as pd
import pytest

from overfit import data
from overfit.config import Config


def _frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "time": np.arange(n) * 86400,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": rng.uniform(1, 10, n),
        }
    )


def _payload(times):
    n = len(times)
    return {"s": "ok", "t": list(times), "o": [1] * n, "h": [1] * n, "l": [1] * n, "c": [1] * n, "v": [1] * n}


def test_window_bounds_cover_range_without_gaps():
    step = 86400
    windows = list(data.window_bounds(0, 1000 * step, step))
    assert windows[0][0] == 0 and windows[-1][1] == 1000 * step
    for (_, prev_hi), (lo, _) in zip(windows, windows[1:]):
        assert lo == prev_hi
    assert all((hi - lo) // step <= data.WINDOW_CANDLES for lo, hi in windows)


def test_parse_udf_ok_no_data_and_error():
    frame = data.parse_udf(_payload([1, 2]))
    assert list(frame.columns) == data.COLUMNS and len(frame) == 2
    assert data.parse_udf({"s": "no_data"}).empty
    with pytest.raises(data.DataError, match="bad symbol"):
        data.parse_udf({"s": "error", "errmsg": "bad symbol"})
    with pytest.raises(data.DataError):
        data.parse_udf({"s": "ok", "t": [1]})


def test_build_dataset_alignment_and_no_nan():
    frame = _frame()
    x, y = data.build_dataset(frame)
    close = frame["close"].to_numpy()
    logret = np.diff(np.log(close))  # logret[i-1] is the return into row i
    first = 10  # first row with a full 10-step rolling window
    assert x.shape == (len(frame) - first - 1, len(data.FEATURES)) and len(y) == len(x)
    assert not np.isnan(x).any()
    np.testing.assert_allclose(x[:, 0], logret[first - 1 : -1], rtol=1e-4)
    expected = (close[first + 1 :] > close[first:-1]).astype(np.float32)
    np.testing.assert_array_equal(y, expected)


def test_build_dataset_rejects_non_positive_close():
    frame = _frame()
    frame.loc[3, "close"] = 0
    with pytest.raises(ValueError):
        data.build_dataset(frame)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        step = data.STEP_SECONDS[params["resolution"]]
        times = range(params["from"], params["to"] + 1, step)
        return _Response(_payload(times))


def test_fetch_merges_windows_and_dedupes():
    cfg = Config(start="2021-01-01", end="2023-01-01", request_pause=0)
    session = _Session()
    frame = data.fetch_ohlcv(cfg, session=session)
    assert len(session.calls) > 1
    assert frame["time"].is_unique and frame["time"].is_monotonic_increasing
    assert frame["time"].diff().dropna().eq(86400).all()


def test_fetch_rejects_unknown_resolution():
    with pytest.raises(ValueError):
        data.fetch_ohlcv(Config(resolution="7"))


def test_load_ohlcv_uses_cache(tmp_path, monkeypatch):
    cfg = Config(cache_dir=tmp_path)
    calls = []

    def fake_fetch(_cfg):
        calls.append(1)
        return _frame()

    monkeypatch.setattr(data, "fetch_ohlcv", fake_fetch)
    first = data.load_ohlcv(cfg)
    second = data.load_ohlcv(cfg)
    assert len(calls) == 1 and len(first) == len(second)
    data.load_ohlcv(cfg, refresh=True)
    assert len(calls) == 2
