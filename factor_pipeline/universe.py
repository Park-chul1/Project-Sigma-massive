"""Historical NASDAQ membership and the mandatory market-cap floor."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


WRDS_SOURCE = "compustat_wrds"
BOTTOM_MARKET_CAP_FRACTION = 0.20


def validate_wrds_source(frame: pd.DataFrame, label: str = "data") -> None:
    if frame.empty:
        raise ValueError(f"{label} is empty; rebuild from Compustat/WRDS")
    if "source" not in frame or not frame["source"].eq(WRDS_SOURCE).all():
        raise ValueError(f"{label} must come exclusively from Compustat/WRDS; rebuild the dataset")


def build_nasdaq_membership_mask(
    membership: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str],
) -> pd.DataFrame:
    """Union of Compustat historical EXCHG=14 intervals (inclusive bounds).

    Re-entry is supported; current exchange/status is never backfilled into
    earlier dates. A missing through date represents a still-open interval.
    """
    required = {"ticker", "from_date", "thru_date"}
    if not required.issubset(membership):
        raise ValueError(f"NASDAQ membership missing columns: {sorted(required - set(membership))}")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Model security identifiers must be unique")
    dates = pd.DatetimeIndex(dates).normalize()
    result = pd.DataFrame(False, index=dates, columns=tickers)
    intervals = membership.copy()
    intervals["from_date"] = pd.to_datetime(intervals["from_date"], errors="raise")
    intervals["thru_date"] = pd.to_datetime(intervals["thru_date"], errors="raise")
    if intervals["from_date"].isna().any() or intervals["ticker"].isna().any():
        raise ValueError("NASDAQ membership requires security identifiers and start dates")
    if (intervals["thru_date"] < intervals["from_date"]).any():
        raise ValueError("NASDAQ membership has a reversed date interval")
    for row in intervals.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker not in result:
            continue
        active = dates >= row.from_date
        if pd.notna(row.thru_date):
            active &= dates <= row.thru_date
        result.loc[active, ticker] = True
    return result


def exclude_bottom_market_cap(market_cap: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Keep the top 80% by each day's positive, observed market cap.

    Missing/nonpositive capitalization is excluded, never filled from a future
    observation. Round the number removed up so at least 20% is excluded.
    Break equal-cap ties by stable security identifier, independent of columns.
    Apply other trading filters *after* this function to keep the ranking
    population equal to that day's NASDAQ members.
    """
    caps = market_cap.reindex(index=eligible.index, columns=eligible.columns).astype(float)
    valid = eligible.fillna(False).astype(bool) & np.isfinite(caps) & caps.gt(0)
    kept = pd.DataFrame(False, index=eligible.index, columns=eligible.columns)
    for date in eligible.index:
        names = sorted(valid.columns[valid.loc[date]], key=str)
        ranked = caps.loc[date, names].sort_values(kind="stable")
        removed = math.ceil(len(ranked) * BOTTOM_MARKET_CAP_FRACTION)
        kept.loc[date, ranked.index[removed:]] = True
    return kept


def load_wrds_universe_mask(input_dir: str | Path, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Revalidate stored universe inputs instead of accepting a legacy mask."""
    root = Path(input_dir)
    membership = pd.read_parquet(root / "nasdaq_membership.parquet")
    bars = pd.read_parquet(root / "daily_bars.parquet")
    validate_wrds_source(membership, "NASDAQ membership")
    validate_wrds_source(bars, "Daily bars")
    if "market_cap" not in bars:
        raise ValueError("Daily WRDS bars must include point-in-time market_cap")
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    caps = bars.pivot(index="date", columns="ticker", values="market_cap")
    members = build_nasdaq_membership_mask(membership, dates, tickers)
    return exclude_bottom_market_cap(caps, members)
