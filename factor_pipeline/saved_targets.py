"""Validated WRDS model targets and raw closing prices for broker adapters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factor_pipeline.signals import make_quantile_long_short_weights, make_scores, predict_factor_returns
from factor_pipeline.universe import load_wrds_universe_mask, validate_wrds_source


def load_pipeline_target_weights(
    path: Path, date_index: int = -1, method: str = "ewma", lookback: int = 20,
    ewma_halflife: float = 20.0, min_periods: int = 5, quantile: float = 0.10,
    gross: float = 2.0,
) -> tuple[list[str], np.ndarray]:
    from live.data import load_broker_symbol_map

    if method == "oracle":
        raise ValueError("Oracle predictions cannot be used for broker targets")
    metadata = pd.read_csv(path / "tickers.csv", dtype={"ticker": str, "symbol": str})
    validate_wrds_source(metadata, "Security metadata")
    if not {"ticker", "symbol", "delisted_utc"}.issubset(metadata):
        raise ValueError("WRDS metadata requires stable IDs, symbols, and delisted_utc")
    tickers = metadata["ticker"].tolist()
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path / "dates.csv")["date"]))
    eligible = load_wrds_universe_mask(path, dates, tickers).to_numpy(dtype=bool)
    saved_mask = np.load(path / "tradable_mask.npy")
    if eligible.shape != saved_mask.shape:
        raise ValueError("Saved tradable mask does not align with dates and securities")
    eligible &= saved_mask.astype(bool)
    if "delisted_utc" in metadata:
        for i, event in enumerate(pd.to_datetime(metadata["delisted_utc"], errors="coerce")):
            if pd.notna(event):
                eligible[dates >= event, i] = False
    if not -len(dates) <= date_index < len(dates):
        raise IndexError("date_index is outside the saved history")
    date_index %= len(dates)

    saved = next((path / name for name in ("weights.npy", "positions.npy") if (path / name).exists()), None)
    if saved is not None:
        weights = np.load(saved)
        if weights.shape != eligible.shape:
            raise ValueError("Saved weights do not align with dates and securities")
        target = np.where(eligible[date_index], weights[date_index], 0.0)
    else:
        scores_path = path / "scores.npy"
        if scores_path.exists():
            scores = np.load(scores_path)
        else:
            predictions = predict_factor_returns(
                np.load(path / "factor_returns.npy"), method=method, lookback=lookback,
                ewma_halflife=ewma_halflife, min_periods=min_periods,
            )
            scores = make_scores(np.load(path / "X.npy"), predictions, tradable_mask=eligible)
        if scores.shape != eligible.shape:
            raise ValueError("Saved scores do not align with dates and securities")
        target = make_quantile_long_short_weights(
            scores[date_index], eligible[date_index], quantile=quantile, gross=gross,
        )
    selected = np.flatnonzero(np.isfinite(target) & (target != 0))
    ids = [tickers[i] for i in selected]
    symbols = load_broker_symbol_map(path / "tickers.csv", ids)
    return [str(symbols.loc[ticker]) for ticker in ids], target[selected]


def load_wrds_closing_prices(path: Path, symbols: list[str], date_index: int = -1) -> dict[str, float]:
    """Use the chosen signal date's raw close; never a future/adjusted quote."""
    dates = pd.to_datetime(pd.read_csv(path / "dates.csv")["date"])
    asof = dates.iloc[date_index].normalize()
    bars = pd.read_parquet(path / "daily_bars.parquet")
    metadata = pd.read_csv(path / "tickers.csv", dtype={"ticker": str, "symbol": str})
    validate_wrds_source(bars, "Daily bars")
    validate_wrds_source(metadata, "Security metadata")
    bars = bars[pd.to_datetime(bars["date"]).dt.normalize().eq(asof)]
    if not {"ticker", "symbol", "delisted_utc"}.issubset(metadata):
        raise ValueError("WRDS metadata requires stable IDs, symbols, and delisted_utc")
    tickers = metadata["ticker"].tolist()
    eligible = load_wrds_universe_mask(path, pd.DatetimeIndex([asof]), tickers).iloc[0]
    events = pd.to_datetime(metadata["delisted_utc"], errors="coerce")
    selected = metadata.loc[
        metadata["symbol"].isin(symbols) & eligible.to_numpy() & (events.isna() | events.gt(asof))
    ]
    if selected["symbol"].duplicated().any():
        raise ValueError("Ambiguous broker symbols in WRDS security metadata")
    prices = selected[["ticker", "symbol"]].merge(bars[["ticker", "raw_close"]], on="ticker", how="left", validate="one_to_one")
    if prices["symbol"].duplicated().any():
        raise ValueError("Ambiguous broker symbols in WRDS prices")
    out = prices.set_index("symbol")["raw_close"].to_dict()
    missing = [symbol for symbol in symbols if symbol not in out or not np.isfinite(out[symbol]) or out[symbol] <= 0]
    if missing:
        raise ValueError(f"Missing WRDS raw close on {asof.date()} for {missing}")
    return out
