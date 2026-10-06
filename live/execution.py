from __future__ import annotations

import math
import pandas as pd

from live.broker import Order
from live.config import LiveConfig


def latest_prices_from_bars(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    if "raw_close" not in bars:
        raise ValueError("Broker share sizing requires Compustat raw_close; adjusted close is not an execution price")
    return bars.drop_duplicates("ticker", keep="last").set_index("ticker")["raw_close"].astype(float)


def current_weights_from_positions(positions: pd.Series, prices: pd.Series, equity: float) -> pd.Series:
    if positions.empty or equity <= 0:
        return pd.Series(dtype=float)
    idx = positions.index.intersection(prices.index)
    values = positions.reindex(idx).astype(float) * prices.reindex(idx).astype(float)
    return (values / equity).replace([float("inf"), float("-inf")], pd.NA).dropna()


def generate_orders(
    current_positions: pd.Series,
    target_weights: pd.Series,
    prices: pd.Series,
    equity: float,
    config: LiveConfig,
) -> list[Order]:
    orders: list[Order] = []
    symbols = current_positions.index.union(target_weights.index)
    for ticker in symbols:
        price = float(prices.get(ticker, float("nan")))
        if not math.isfinite(price) or price <= 0:
            continue
        current_qty = int(current_positions.get(ticker, 0))
        target_dollars = float(target_weights.get(ticker, 0.0)) * equity
        target_qty = int(math.trunc(target_dollars / price))
        diff = target_qty - current_qty
        if diff == 0:
            continue
        notional = abs(diff * price)
        if notional < config.MIN_ORDER_DOLLARS:
            continue
        side = "BUY" if diff > 0 else "SELL"
        pad = 1.0 + (config.SLIPPAGE_BPS / 10_000.0 if side == "BUY" else -config.SLIPPAGE_BPS / 10_000.0)
        orders.append(Order(ticker=ticker, side=side, quantity=abs(diff), limit_price=round(price * pad, 4), dry_run=config.DRY_RUN))
    return orders
