from __future__ import annotations

import numpy as np
import pandas as pd

from factor_pipeline.signals import (
    predict_factor_returns,
    make_scores,
    make_positions_from_scores,
)


def compute_turnover(weights: np.ndarray) -> np.ndarray:
    """One-way turnover: 0.5 * sum(abs(w_t - w_{t-1}))."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D [T,N], got shape {weights.shape}")
    turnover = np.full(weights.shape[0], np.nan, dtype=float)
    if weights.shape[0] == 0:
        return turnover
    turnover[0] = 0.5 * np.nansum(np.abs(weights[0]))
    for t in range(1, weights.shape[0]):
        turnover[t] = 0.5 * np.nansum(np.abs(weights[t] - weights[t - 1]))
    return turnover


def portfolio_returns(
    weights: np.ndarray,
    r: np.ndarray,
    min_return_coverage: float = 0.80,
    max_abs_return: float | None = 1.0,
    terminal_return_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute daily portfolio returns from weights[t] and forward returns r[t].

    Missing future returns are not known at trading time. For this first research
    backtest, a date is kept only if at least min_return_coverage of active
    positions have finite next-period returns. Missing returns among the kept
    dates contribute zero PnL because weights are not renormalized ex-post.
    """
    weights = np.asarray(weights, dtype=float)
    r = np.asarray(r, dtype=float)
    if weights.shape != r.shape:
        raise ValueError(f"weights shape {weights.shape} must equal r shape {r.shape}")
    terminal = np.zeros(r.shape, dtype=bool)
    if terminal_return_mask is not None:
        terminal = np.asarray(terminal_return_mask, dtype=bool)
        if terminal.shape != r.shape:
            raise ValueError("terminal_return_mask must align with returns")
    if max_abs_return is not None:
        if max_abs_return <= 0:
            raise ValueError("max_abs_return must be positive")
        r = np.where((np.abs(r) <= max_abs_return) | terminal, r, np.nan)

    T = weights.shape[0]
    port = np.full(T, np.nan, dtype=float)
    coverage = np.full(T, np.nan, dtype=float)
    for t in range(T):
        active = np.abs(weights[t]) > 0
        n_active = int(active.sum())
        if n_active == 0:
            continue
        finite_ret = np.isfinite(r[t])
        cov = float((active & finite_ret).sum() / n_active)
        coverage[t] = cov
        if cov < min_return_coverage:
            continue
        port[t] = float(np.nansum(weights[t] * np.where(finite_ret, r[t], 0.0)))
    return port, coverage


def summarize_backtest(
    returns: np.ndarray,
    weights: np.ndarray,
    turnover: np.ndarray,
    periods_per_year: int = 252,
) -> dict:
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    turnover = np.asarray(turnover, dtype=float)
    valid = np.isfinite(returns)
    if not valid.any():
        return {
            "n_periods": 0,
            "mean_return": None,
            "volatility": None,
            "sharpe": None,
            "total_return": None,
            "max_drawdown": None,
            "win_rate": None,
            "avg_gross_exposure": float(np.nanmean(np.nansum(np.abs(weights), axis=1))) if weights.size else None,
            "avg_turnover": float(np.nanmean(turnover)) if turnover.size else None,
        }

    ret = returns[valid]
    mean = float(np.nanmean(ret))
    vol = float(np.nanstd(ret, ddof=1)) if ret.size > 1 else np.nan
    sharpe = float(np.sqrt(periods_per_year) * mean / vol) if np.isfinite(vol) and vol > 1e-12 else None
    equity = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    gross = np.nansum(np.abs(weights), axis=1)
    return {
        "n_periods": int(ret.size),
        "mean_return": mean,
        "volatility": vol if np.isfinite(vol) else None,
        "sharpe": sharpe,
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(np.nanmin(drawdown)),
        "win_rate": float(np.mean(ret > 0)),
        "avg_gross_exposure": float(np.nanmean(gross)),
        "avg_turnover": float(np.nanmean(turnover)),
    }


def run_factor_backtest(
    X: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    tradable_mask: np.ndarray | None = None,
    method: str = "latest",
    lookback: int = 20,
    ewma_halflife: float = 20.0,
    min_periods: int = 5,
    quantile: float = 0.10,
    gross: float = 2.0,
    min_names_per_side: int = 5,
    min_return_coverage: float = 0.80,
    max_abs_return: float | None = 1.0,
    terminal_return_mask: np.ndarray | None = None,
) -> dict:
    f_pred = predict_factor_returns(
        f,
        method=method,
        lookback=lookback,
        ewma_halflife=ewma_halflife,
        min_periods=min_periods,
    )
    scores = make_scores(X, f_pred, tradable_mask=tradable_mask)
    weights = make_positions_from_scores(
        scores,
        tradable_mask=tradable_mask,
        quantile=quantile,
        gross=gross,
        min_names_per_side=min_names_per_side,
    )
    returns, return_coverage = portfolio_returns(
        weights,
        r,
        min_return_coverage=min_return_coverage,
        max_abs_return=max_abs_return,
        terminal_return_mask=terminal_return_mask,
    )
    turnover = compute_turnover(weights)
    metrics = summarize_backtest(returns, weights, turnover)

    n_long = (weights > 0).sum(axis=1)
    n_short = (weights < 0).sum(axis=1)
    daily = pd.DataFrame({
        "portfolio_return": returns,
        "cumulative_return": np.where(
            np.isfinite(returns),
            np.cumprod(1.0 + np.where(np.isfinite(returns), returns, 0.0)) - 1.0,
            np.nan,
        ),
        "turnover": turnover,
        "gross_exposure": np.nansum(np.abs(weights), axis=1),
        "n_long": n_long,
        "n_short": n_short,
        "return_coverage": return_coverage,
    })

    return {
        "f_pred": f_pred,
        "scores": scores,
        "weights": weights,
        "returns": returns,
        "turnover": turnover,
        "daily": daily,
        "metrics": metrics,
    }
