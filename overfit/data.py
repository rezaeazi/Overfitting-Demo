"""Nobitex OHLCV download (cached on disk) and feature engineering."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

log = logging.getLogger(__name__)

BASE_URL = "https://apiv2.nobitex.ir/market/udf/history"
STEP_SECONDS = {"60": 3600, "D": 86400}
WINDOW_CANDLES = 400  # stay below the API cap of 500 candles per request
TIMEOUT = 10
COLUMNS = ["time", "open", "high", "low", "close", "volume"]
FEATURES = ("ret_1", "ret_5", "ret_10", "vol_10")  # volume is 0 in early Nobitex candles


class DataError(RuntimeError):
    """Raised when the exchange returns an unusable response."""


def window_bounds(start: int, end: int, step: int) -> Iterator[tuple[int, int]]:
    """Yield [from, to] unix windows; neighbours share one boundary candle."""
    span = WINDOW_CANDLES * step
    for lo in range(start, end, span):
        yield lo, min(lo + span, end)


def parse_udf(payload: dict) -> pd.DataFrame:
    """Convert a TradingView-UDF payload into an OHLCV frame."""
    status = payload.get("s")
    if status == "no_data":
        return pd.DataFrame(columns=COLUMNS)
    if status != "ok":
        raise DataError(payload.get("errmsg") or f"unexpected status: {status!r}")
    try:
        frame = pd.DataFrame(
            {
                "time": payload["t"],
                "open": payload["o"],
                "high": payload["h"],
                "low": payload["l"],
                "close": payload["c"],
                "volume": payload["v"],
            }
        )
    except KeyError as exc:
        raise DataError(f"missing field in response: {exc}") from exc
    return frame.astype({c: "float64" for c in COLUMNS[1:]} | {"time": "int64"})


def _to_unix(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def _retrying_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_ohlcv(cfg: Config, session: requests.Session | None = None) -> pd.DataFrame:
    """Download the full configured range, one window per request."""
    if cfg.resolution not in STEP_SECONDS:
        raise ValueError(f"resolution must be one of {sorted(STEP_SECONDS)}")
    session = session or _retrying_session()
    step = STEP_SECONDS[cfg.resolution]
    end = _to_unix(cfg.end) if cfg.end else int(time.time())
    frames = []
    for lo, hi in window_bounds(_to_unix(cfg.start), end, step):
        params = {"symbol": cfg.symbol, "resolution": cfg.resolution, "from": lo, "to": hi}
        response = session.get(BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        frame = parse_udf(response.json())
        if not frame.empty:
            frames.append(frame)
        time.sleep(cfg.request_pause)
    if not frames:
        raise DataError(f"no candles returned for {cfg.symbol} ({cfg.resolution})")
    return (
        pd.concat(frames)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )


def load_ohlcv(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Return cached candles, downloading them first when needed."""
    path = Path(cfg.cache_dir) / f"{cfg.symbol}_{cfg.resolution}.csv"
    if not path.exists() or refresh:
        frame = fetch_ohlcv(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(path)  # atomic: an interrupted run never leaves a corrupt cache
        log.info("cached %d candles to %s", len(frame), path)
    frame = pd.read_csv(path)
    return frame[frame["time"] + STEP_SECONDS[cfg.resolution] <= time.time()]  # drop the open candle


def build_dataset(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised features and next-candle direction labels (1.0 = up).

    Row i holds features known at close i; its label is the sign of the
    return from close i to close i+1, so no feature looks into the future.
    """
    if (frame["close"] <= 0).any():
        raise ValueError("close prices must be positive")
    logret = np.log(frame["close"]).diff()
    features = pd.DataFrame(
        {
            "ret_1": logret,
            "ret_5": logret.rolling(5).sum(),
            "ret_10": logret.rolling(10).sum(),
            "vol_10": logret.rolling(10).std(),
        }
    )
    next_ret = logret.shift(-1)
    valid = features.notna().all(axis=1) & next_ret.notna()
    x = features.loc[valid, list(FEATURES)].to_numpy(dtype=np.float32)
    y = (next_ret[valid] > 0).to_numpy(dtype=np.float32)
    return x, y
