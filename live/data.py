from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from factor_pipeline.panel import delisting_dates
from factor_pipeline.universe import (
    build_nasdaq_membership_mask,
    exclude_bottom_market_cap,
    validate_wrds_source,
)
from factor_pipeline.wrds_client import WRDSClient, download_daily_bars, download_nasdaq_membership
from live.config import LiveConfig

BAR_COLUMNS = [
    "timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions",
    "raw_close", "market_cap", "source",
]


def naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def completed_daily_asof(value=None) -> pd.Timestamp:
    """Use only completed New York calendar days; WRDS is a daily feed."""
    if value is not None:
        return naive_timestamp(value).normalize()
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None) - pd.Timedelta(days=1)


@dataclass
class FetchResult:
    bars: pd.DataFrame
    latency_seconds: float
    symbols_requested: int
    symbols_updated: int
    asof_time: pd.Timestamp


class CompustatDailyProvider:
    """Compustat security daily prices accessed exclusively through WRDS."""

    def __init__(self, config: LiveConfig):
        config.validate()
        self.config = config
        self.client = WRDSClient(
            wrds_username=config.WRDS_USERNAME,
            cache_dir=config.WRDS_CACHE_PATH,
            use_cache=True,
        )

    def _fetch(self, universe: list[str], start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        started = time.perf_counter()
        membership = download_nasdaq_membership(self.client, start.date().isoformat(), end.date().isoformat())
        if universe:
            membership = membership[membership["ticker"].isin(universe)].copy()
        if membership.empty:
            raise RuntimeError("WRDS returned no NASDAQ membership for the requested daily data window")
        bars = normalize_bars(download_daily_bars(self.client, membership, start.date().isoformat(), end.date().isoformat()))
        if bars.empty:
            raise RuntimeError("WRDS returned no Compustat daily bars; verify data coverage and WRDS access")
        validate_wrds_source(bars, "live daily bars")
        return FetchResult(bars, time.perf_counter() - started, len(universe), int(bars["ticker"].nunique()), end)

    def fetch_latest_daily_bars(self, universe: list[str], asof_time=None) -> FetchResult:
        end = completed_daily_asof(asof_time)
        return self._fetch(universe, end - pd.Timedelta(days=7), end)

    def prefetch_history(self, universe: list[str], asof_time=None, limit: int | None = None) -> FetchResult:
        end = completed_daily_asof(asof_time)
        symbols = universe if limit is None else universe[:limit]
        return self._fetch(symbols, end - pd.Timedelta(days=self.config.WRDS_PREFETCH_DAYS), end)

    def disconnect(self) -> None:
        self.client.close()


class CachedOnlyProvider:
    def __init__(self, config: LiveConfig):
        self.config = config

    def fetch_latest_daily_bars(self, universe: list[str], asof_time=None) -> FetchResult:
        started = time.perf_counter()
        end = completed_daily_asof(asof_time)
        history = load_recent_history(universe, self.config.LOOKBACK_BARS, self.config.CACHE_PATH, end_time=end)
        if history.empty:
            raise RuntimeError("No WRDS daily bars in cache; run the WRDS pipeline before using cache mode")
        bars = history[history["timestamp"].eq(history["timestamp"].max())].copy()
        return FetchResult(bars, time.perf_counter() - started, len(universe), int(bars["ticker"].nunique()), end)


def make_provider(config: LiveConfig) -> CompustatDailyProvider | CachedOnlyProvider:
    config.validate()
    if config.DATA_PROVIDER.lower() == "wrds":
        return CompustatDailyProvider(config)
    if config.DATA_PROVIDER.lower() == "cache":
        return CachedOnlyProvider(config)
    raise ValueError("DATA_PROVIDER must be wrds or cache; all market data must originate in Compustat WRDS")


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    out = bars.copy()
    if "timestamp" not in out.columns and "date" in out.columns:
        out["timestamp"] = out["date"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    out["ticker"] = out["ticker"].astype(str)
    for col in BAR_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
        if col not in {"timestamp", "ticker", "source"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[BAR_COLUMNS].dropna(subset=["timestamp", "ticker"])
    return out.drop_duplicates(["timestamp", "ticker"], keep="last").sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def update_local_bar_cache(bars: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    bars = normalize_bars(bars)
    if not bars.empty:
        validate_wrds_source(bars, "incoming daily bars")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        old = normalize_bars(pd.read_parquet(cache_path))
        if not old.empty:
            validate_wrds_source(old, "existing daily cache")
        combined = pd.concat([old, bars], ignore_index=True) if not bars.empty else old
    else:
        combined = bars
    combined = combined.drop_duplicates(["timestamp", "ticker"], keep="last").sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    combined.to_parquet(cache_path, index=False)
    return combined


def load_recent_history(universe: list[str], lookback: int, cache_path: Path, end_time=None) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing WRDS daily cache: {cache_path}")
    frame = normalize_bars(pd.read_parquet(cache_path))
    validate_wrds_source(frame, "daily history cache")
    if universe:
        frame = frame[frame["ticker"].isin(universe)]
    if end_time is not None:
        frame = frame[frame["timestamp"] <= naive_timestamp(end_time)]
    times = sorted(frame["timestamp"].dropna().unique())[-lookback:]
    return frame[frame["timestamp"].isin(times)].sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def load_universe(config: LiveConfig, asof_time=None) -> list[str]:
    """Historical NASDAQ membership, excluding the smallest 20% by market cap."""
    config.validate()
    asof = completed_daily_asof(asof_time)
    for path in [config.MEMBERSHIP_PATH, config.UNIVERSE_BARS_PATH, config.UNIVERSE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing WRDS universe input: {path}; run scripts/run_clean_pipeline.py first")
    membership = pd.read_parquet(config.MEMBERSHIP_PATH)
    bars = pd.read_parquet(config.UNIVERSE_BARS_PATH)
    metadata_path = config.UNIVERSE_PATH
    metadata = pd.read_parquet(metadata_path) if metadata_path.suffix == ".parquet" else pd.read_csv(metadata_path, dtype={"ticker": str})
    validate_wrds_source(metadata, "universe security metadata")
    if not {"ticker", "delisted_utc"}.issubset(metadata):
        raise ValueError("WRDS security metadata must contain ticker and delisted_utc")
    validate_wrds_source(membership, "NASDAQ membership")
    validate_wrds_source(bars, "universe daily bars")
    if "market_cap" not in bars:
        raise ValueError("WRDS daily bars must include market_cap; no alternative ranking is permitted")
    date_column = "date" if "date" in bars else "timestamp"
    bars["date"] = pd.to_datetime(bars[date_column]).dt.normalize()
    bars = bars[bars["date"].le(asof)]
    if bars.empty:
        raise RuntimeError(f"No WRDS market-cap observations available as of {asof.date()}")
    latest = bars["date"].max()
    if (asof - latest).days > 7:
        raise RuntimeError(f"WRDS universe cache is stale: latest={latest.date()}, requested={asof.date()}")
    bars = bars[bars["date"].eq(latest)].drop_duplicates("ticker", keep="last").set_index("ticker")
    tickers = sorted(membership["ticker"].dropna().astype(str).unique())
    if set(tickers) - set(metadata["ticker"]):
        raise ValueError("WRDS security metadata does not cover the NASDAQ membership universe")
    dates = pd.DatetimeIndex([asof])
    caps = pd.DataFrame([pd.to_numeric(bars["market_cap"].reindex(tickers), errors="coerce").to_numpy()], index=dates, columns=tickers)
    eligible = build_nasdaq_membership_mask(membership, dates, tickers)
    keep = exclude_bottom_market_cap(caps, eligible)
    ended = delisting_dates(metadata, tickers).le(asof)
    keep.loc[:, ended] = False
    selected = caps.loc[asof, np.asarray(keep)[0]].sort_values(ascending=False, kind="stable")
    if selected.empty:
        raise RuntimeError("No NASDAQ securities remain after the mandatory market-cap exclusion")
    return selected.index.tolist()[:config.UNIVERSE_SIZE_LIMIT]


def load_broker_symbol_map(metadata_path: Path, model_ids: list[str]) -> pd.Series:
    """Require an unambiguous Compustat identifier to executable symbol mapping."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing WRDS security metadata for broker symbol mapping: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"ticker": str, "symbol": str})
    validate_wrds_source(frame, "broker symbol metadata")
    if not {"ticker", "symbol"}.issubset(frame.columns):
        raise ValueError("WRDS metadata must contain ticker (stable security ID) and symbol")
    selected = frame[frame["ticker"].isin(model_ids)][["ticker", "symbol"]].copy()
    if selected["symbol"].isna().any() or selected["ticker"].duplicated().any():
        raise ValueError("Missing or ambiguous WRDS broker symbol mapping")
    selected["symbol"] = selected["symbol"].str.strip()
    if selected["symbol"].eq("").any() or selected["symbol"].duplicated().any():
        raise ValueError("Missing or ambiguous WRDS broker symbol mapping")
    mapping = selected.set_index("ticker")["symbol"].reindex(model_ids)
    if mapping.isna().any():
        raise ValueError(f"No WRDS broker symbol mapping for: {mapping[mapping.isna()].index.tolist()}")
    return mapping
