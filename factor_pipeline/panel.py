from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "adj_close", "volume", "vwap", "transactions", "market_cap", "raw_open", "raw_close", "raw_volume"]
DELISTING_RECOVERY_FRACTION = 0.5


def delisting_dates(ticker_metadata: pd.DataFrame | None, tickers: list[str]) -> pd.Series:
    """Read confirmed security delistings, never infer one from a missing quote."""
    result = pd.Series(pd.NaT, index=tickers, dtype="datetime64[ns]")
    if ticker_metadata is None or ticker_metadata.empty or "delisted_utc" not in ticker_metadata:
        return result
    if "ticker" not in ticker_metadata:
        raise ValueError("Delisting metadata must contain ticker")
    meta = ticker_metadata.copy()
    meta["ticker"] = meta["ticker"].astype(str)
    if meta["ticker"].duplicated().any():
        raise ValueError("Delisting metadata must have one row per security")
    values = pd.to_datetime(meta.set_index("ticker")["delisted_utc"], errors="coerce", utc=True)
    return values.dt.tz_convert(None).dt.normalize().reindex(tickers)


def bars_long_to_panel(bars: pd.DataFrame, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    if bars.empty:
        raise ValueError("bars is empty")
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    if tickers is None:
        tickers = sorted(bars["ticker"].dropna().unique().tolist())
    dates = pd.DatetimeIndex(sorted(bars["date"].unique()), name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in FIELDS:
        src = "close" if field == "adj_close" and "adj_close" not in bars.columns else field
        if src not in bars.columns:
            continue
        wide = bars.pivot_table(index="date", columns="ticker", values=src, aggfunc="last")
        wide = wide.reindex(index=dates, columns=tickers).astype(float)
        panel[field] = wide
    return panel


def build_tradable_mask(
    panel: dict[str, pd.DataFrame],
    require_volume: bool = True,
) -> pd.DataFrame:
    """Return a date x ticker mask for names tradable on each date.

    The mask is intentionally derived from the historical daily bars already
    loaded into the pipeline, so it adds only one boolean T x N array and does
    not require per-ticker state objects.
    """
    if "adj_close" in panel:
        price = panel["adj_close"]
    elif "close" in panel:
        price = panel["close"]
    else:
        raise KeyError("panel must contain adj_close or close")

    price_arr = price.to_numpy(dtype=float)
    mask = pd.DataFrame(
        np.isfinite(price_arr) & (price_arr > 0),
        index=price.index,
        columns=price.columns,
    )

    if require_volume and "volume" in panel:
        volume = panel["volume"].reindex_like(price)
        volume_arr = volume.to_numpy(dtype=float)
        volume_mask = pd.DataFrame(
            np.isfinite(volume_arr) & (volume_arr > 0),
            index=price.index,
            columns=price.columns,
        )
        mask &= volume_mask

    return mask.astype(bool)


def compute_forward_returns(
    adj_close: pd.DataFrame,
    horizon: int = 1,
    max_abs_return: float | None = None,
    ticker_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    prices = adj_close.where(np.isfinite(adj_close) & adj_close.gt(0))
    returns = prices.shift(-horizon) / prices - 1.0
    settlements = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    dates = pd.DatetimeIndex(prices.index).normalize()
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Price dates must be strictly increasing")
    events = delisting_dates(ticker_metadata, list(prices.columns))
    for ticker, event in events.dropna().items():
        # All horizons crossing a known event settle once, even if stale quotes
        # happen to remain in the source after the security ceased trading.
        returns.loc[dates >= event, ticker] = np.nan
        before = prices.loc[dates < event, ticker].dropna()
        if before.empty:
            continue
        recovery = DELISTING_RECOVERY_FRACTION * float(before.iloc[-1])
        if horizon < len(dates):
            starts = np.flatnonzero((dates[:-horizon] < event) & (dates[horizon:] >= event))
            column = prices.columns.get_loc(ticker)
            returns.iloc[starts, column] = recovery / prices.iloc[starts, column] - 1.0
            settlements.iloc[starts, column] = True
    if max_abs_return is not None:
        if max_abs_return <= 0:
            raise ValueError("max_abs_return must be positive")
        returns = returns.mask((returns.abs() > max_abs_return) & ~settlements)
    return returns


def finite_ratio(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0
