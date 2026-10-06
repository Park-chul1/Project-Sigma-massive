from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from backtests.close_to_next_open_daily import (
    DailyBacktestConfig,
    config_from_dict,
    daily_bars_date_range,
    execution_bars_path,
    write_residual_heatmap,
    _rank_weights_one_day,
    _safe_git_hash,
)
from factor_pipeline.simple_config import load_config_dict
from factor_pipeline.signals import predict_factor_returns
from factor_pipeline.panel import delisting_dates
from factor_pipeline.universe import load_wrds_universe_mask, validate_wrds_source


def load_daily_job_config(path: str | Path | None) -> DailyBacktestConfig:
    return config_from_dict(load_config_dict(path))


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _assert_signal_date_available(cfg: DailyBacktestConfig, signal_date: pd.Timestamp) -> None:
    dates_path = Path(cfg.input_dir) / "dates.csv"
    if not dates_path.exists():
        raise FileNotFoundError(f"Missing processed dates file: {dates_path}")
    processed_dates = pd.to_datetime(pd.read_csv(dates_path)["date"]).dt.normalize()
    if processed_dates.empty:
        raise RuntimeError(f"No processed dates found in {dates_path}")
    signal_date = pd.Timestamp(signal_date).normalize()
    processed_end = processed_dates.max()
    if signal_date not in set(processed_dates):
        raise RuntimeError(
            f"Refusing after-close job: requested signal_date={signal_date.date()} is not in {dates_path}. "
            f"Processed data range is {processed_dates.min().date()}..{processed_end.date()}."
        )
    if processed_end != signal_date:
        raise RuntimeError(
            f"Refusing after-close job: processed data is not pinned to requested signal_date. "
            f"requested={signal_date.date()} processed_end={processed_end.date()} input_dir={cfg.input_dir}"
        )

    market_bars_path = execution_bars_path(cfg)
    bars_start, bars_end = daily_bars_date_range(market_bars_path)
    if bars_end is None:
        raise RuntimeError(f"Refusing after-close job: execution/daily bars path is missing or empty: {market_bars_path}")
    if pd.Timestamp(bars_end) < signal_date:
        raise RuntimeError(
            f"Refusing after-close job: daily bars do not include requested signal_date close. "
            f"requested={signal_date.date()} bars_range={bars_start}..{bars_end} bars_path={market_bars_path}"
        )


def _load_static_inputs(cfg: DailyBacktestConfig) -> dict:
    input_dir = Path(cfg.input_dir)
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(input_dir / "dates.csv")["date"]).dt.normalize())
    tickers_df = pd.read_csv(input_dir / "tickers.csv", dtype={"ticker": str, "symbol": str})
    validate_wrds_source(tickers_df, "daily report security metadata")
    if not {"symbol", "delisted_utc"}.issubset(tickers_df):
        raise ValueError("WRDS security metadata must include broker symbols and delisted_utc")
    ticker_col = "ticker" if "ticker" in tickers_df.columns else tickers_df.columns[0]
    tickers = tickers_df[ticker_col].astype(str).tolist()
    factor_df = pd.read_csv(input_dir / "factor_names.csv")
    factor_col = "factor" if "factor" in factor_df.columns else factor_df.columns[0]
    factor_names = factor_df[factor_col].astype(str).tolist()
    name_col = "name" if "name" in tickers_df.columns else None
    company_names = (
        tickers_df.set_index(ticker_col)[name_col].astype(str).to_dict()
        if name_col is not None
        else {}
    )
    return {
        "dates": dates,
        "tickers": tickers,
        "factor_names": factor_names,
        "company_names": company_names,
        "symbols": tickers_df.set_index(ticker_col)["symbol"].to_dict(),
        "delisting_dates": delisting_dates(tickers_df, tickers),
    }


def _load_signal_bars(path: Path, signal_date: pd.Timestamp, tickers: list[str]) -> pd.DataFrame:
    bars = pd.read_parquet(path)
    validate_wrds_source(bars, "daily report bars")
    if "raw_close" not in bars:
        raise ValueError("WRDS bars require raw_close for broker sizing and price filters")
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars = bars[bars["date"].eq(signal_date)].copy()
    bars["ticker"] = bars["ticker"].astype(str)
    return bars.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(tickers)


def _signal_quantile_buckets(alpha: np.ndarray) -> dict[int, float]:
    finite = np.isfinite(alpha)
    if finite.sum() < 10:
        return {int(i): np.nan for i in np.where(finite)[0]}
    buckets = pd.qcut(pd.Series(alpha[finite]), 10, labels=False, duplicates="drop")
    return dict(zip(np.where(finite)[0], (buckets.astype(float) + 1.0).tolist()))


def _build_one_day_rankings(
    signal_date: pd.Timestamp,
    tickers: list[str],
    company_names: dict[str, str],
    factor_names: list[str],
    X_t: np.ndarray,
    f_hat: np.ndarray,
    alpha: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    dollar_volume: np.ndarray,
    price: np.ndarray,
    cfg: DailyBacktestConfig,
) -> pd.DataFrame:
    contrib = np.where(np.isfinite(X_t), X_t, 0.0) * np.where(np.isfinite(f_hat), f_hat, 0.0)
    ranks = pd.Series(alpha).rank(ascending=False, method="first")
    buckets = _signal_quantile_buckets(alpha)
    rows = []
    for i, ticker in enumerate(tickers):
        c = contrib[i]
        order = np.argsort(np.abs(c))[::-1]
        top_pos = [factor_names[j] for j in order if c[j] > 0][:3]
        top_neg = [factor_names[j] for j in order if c[j] < 0][:3]
        delta = float(target[i] - current[i])
        side = ""
        if delta > 1e-12:
            side = "BUY" if current[i] >= 0 else "COVER"
        elif delta < -1e-12:
            side = "SHORT" if target[i] < 0 else "SELL"
        price_ok = bool(cfg.min_price <= 0 or (np.isfinite(price[i]) and price[i] >= cfg.min_price))
        liquidity_ok = bool(np.isfinite(dollar_volume[i]) and dollar_volume[i] >= cfg.min_dollar_volume)
        risk_ok = bool(abs(target[i]) <= cfg.max_position_weight + 1e-12)
        trading_cost_bps = cfg.transaction_cost_bps + cfg.slippage_bps
        directional_alpha = -alpha[i] if target[i] < 0 else alpha[i]
        net_alpha_after_cost_bps = (directional_alpha * 10_000.0) - trading_cost_bps if np.isfinite(alpha[i]) else np.nan
        passes_cost_threshold = bool(np.isfinite(net_alpha_after_cost_bps) and net_alpha_after_cost_bps >= cfg.min_net_alpha_after_cost_bps)
        reason = ""
        if side:
            if target[i] > 0:
                direction = "top decile"
                main = ", ".join(top_pos)
            elif target[i] < 0:
                direction = "bottom decile"
                main = ", ".join(top_neg)
            else:
                direction = "cost-adjusted no-position band"
                main = ", ".join(top_pos or top_neg)
            main = main or "the cross-sectional factor mix"
            reason = (
                f"{side} because predicted alpha is in the {direction}. "
                f"Net alpha after estimated {trading_cost_bps:.1f} bps trading cost is {net_alpha_after_cost_bps:.2f} bps "
                f"versus threshold {cfg.min_net_alpha_after_cost_bps:.2f} bps. "
                f"Main contributors were {main}. Price filter {'passed' if price_ok else 'failed'}; "
                f"liquidity filter {'passed' if liquidity_ok else 'failed'}."
            )
        rows.append({
            "signal_date": signal_date,
            "ticker": ticker,
            "company_name": company_names.get(ticker, ""),
            "side": side,
            "current_weight": float(current[i]),
            "target_weight": float(target[i]),
            "order_weight_delta": delta,
            "estimated_order_value": delta * cfg.initial_equity,
            "alpha_score": float(alpha[i]) if np.isfinite(alpha[i]) else np.nan,
            "trading_cost_bps": trading_cost_bps,
            "net_alpha_after_cost_bps": net_alpha_after_cost_bps,
            "min_net_alpha_after_cost_bps": cfg.min_net_alpha_after_cost_bps,
            "passes_cost_threshold": passes_cost_threshold,
            "alpha_rank": float(ranks.iloc[i]) if np.isfinite(alpha[i]) else np.nan,
            "quantile_bucket": buckets.get(i, np.nan),
            "top_positive_factor_contributors": ";".join(top_pos),
            "top_negative_factor_contributors": ";".join(top_neg),
            "factor_exposures": json.dumps({factor_names[k]: float(X_t[i, k]) for k in range(len(factor_names)) if np.isfinite(X_t[i, k])}),
            "factor_return_forecast_used": json.dumps({factor_names[k]: float(f_hat[k]) for k in range(len(factor_names)) if np.isfinite(f_hat[k])}),
            "factor_contributions": json.dumps({factor_names[k]: float(c[k]) for k in range(len(factor_names)) if np.isfinite(c[k])}),
            "residual_score": np.nan,
            "outlier_flag": False,
            "price_flag": price_ok,
            "liquidity_flag": liquidity_ok,
            "risk_flag": risk_ok,
            "reason_text": reason,
        })
    return pd.DataFrame(rows)


def _build_factor_report(
    dates: pd.DatetimeIndex,
    signal_idx: int,
    factor_names: list[str],
    f: np.ndarray,
    f_hat: np.ndarray,
) -> pd.DataFrame:
    hist = f[:signal_idx]
    latest_idx = np.where(np.isfinite(hist).any(axis=1))[0]
    latest = hist[latest_idx[-1]] if latest_idx.size else np.full(len(factor_names), np.nan)
    prev_hat = (
        predict_factor_returns(
            f[:signal_idx],
            method="ewma",
            lookback=min(20, max(1, signal_idx)),
            min_periods=1,
        )[-1]
        if signal_idx > 1
        else np.full(len(factor_names), np.nan)
    )
    rolling = hist[-252:] if hist.size else hist
    mean = np.nanmean(rolling, axis=0) if rolling.size else np.full(len(factor_names), np.nan)
    std = np.nanstd(rolling, axis=0) if rolling.size else np.full(len(factor_names), np.nan)
    rows = []
    for k, factor in enumerate(factor_names):
        rows.append({
            "date": dates[signal_idx],
            "factor": factor,
            "latest_estimated_f_t": latest[k],
            "rolling_mean_f": mean[k],
            "kalman_forecast_f_hat": np.nan,
            "previous_day_f_hat": prev_hat[k],
            "day_over_day_change": f_hat[k] - prev_hat[k] if np.isfinite(prev_hat[k]) else np.nan,
            "zscore_factor_return": (latest[k] - mean[k]) / std[k] if np.isfinite(std[k]) and std[k] > 1e-12 else np.nan,
            "kalman_f_hat": np.nan,
            "kalman_gain": np.nan,
            "state_covariance_diag": np.nan,
            "process_noise_q": np.nan,
            "measurement_noise_r": np.nan,
            "kalman_weight": 0.0,
            "rolling_weight": 1.0,
            "final_f_hat": f_hat[k],
        })
    return pd.DataFrame(rows)


def _build_residuals_latest(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    X,
    r,
    f: np.ndarray,
    signal_idx: int,
    threshold: float,
    lookback: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = max(0, signal_idx)
    start = max(0, end - lookback)
    rows = []
    outliers = []
    for t in range(start, end):
        rt = np.asarray(r[t], dtype=float)
        ft = np.asarray(f[t], dtype=float)
        if not np.isfinite(rt).any() or not np.isfinite(ft).any():
            continue
        pred = np.asarray(X[t], dtype=float) @ np.where(np.isfinite(ft), ft, 0.0)
        resid = rt - pred
        mu = np.nanmean(resid)
        sd = np.nanstd(resid)
        z = (resid - mu) / sd if np.isfinite(sd) and sd > 1e-12 else np.full_like(resid, np.nan)
        for i, ticker in enumerate(tickers):
            rows.append({"date": dates[t], "ticker": ticker, "residual": resid[i], "residual_zscore": z[i]})
            if np.isfinite(z[i]) and abs(z[i]) >= threshold:
                outliers.append({
                    "ticker": ticker,
                    "date": dates[t],
                    "residual": resid[i],
                    "residual_zscore": z[i],
                    "side_implication": "positive_outlier" if z[i] > 0 else "negative_outlier",
                    "excluded_from_trading": False,
                    "reason": "Reported only; outlier policy does not trade residual outliers automatically.",
                })
    return pd.DataFrame(rows), pd.DataFrame(outliers)


def _write_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |")
    return "\n".join(lines)


def _write_single_date_report(
    report_dir: Path,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp | None,
    alpha_rankings: pd.DataFrame,
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame,
    factor_report: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    residuals: pd.DataFrame,
    outliers: pd.DataFrame,
    execution_quality: pd.DataFrame,
    risk_summary: dict,
    metadata: dict,
    warnings: list[str],
) -> Path:
    out = report_dir / signal_date.date().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    orders = alpha_rankings[alpha_rankings["side"].ne("")].copy()
    orders.to_csv(out / "orders.csv", index=False)
    target_positions.to_csv(out / "target_positions.csv", index=False)
    current_positions.to_csv(out / "current_positions.csv", index=False)
    factor_report.to_csv(out / "factor_returns.csv", index=False)
    factor_report.to_csv(out / "factor_forecast.csv", index=False)
    alpha_rankings.to_csv(out / "alpha_rankings.csv", index=False)
    quantile_summary.to_csv(out / "quantile_summary.csv", index=False)
    residuals.to_csv(out / "residuals.csv", index=False)
    outliers.to_csv(out / "outliers.csv", index=False)
    execution_quality.to_csv(out / "execution_quality.csv", index=False)
    (out / "risk_summary.json").write_text(json.dumps(risk_summary, indent=2, default=_json_default), encoding="utf-8")
    (out / "pipeline_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    write_residual_heatmap(residuals, out / "residual_heatmap.png")
    write_residual_heatmap(residuals, out / "factor_dashboard.png")

    buys = orders[orders["side"].isin(["BUY", "COVER"])].sort_values("alpha_score", ascending=False).head(10)
    sells = orders[orders["side"].isin(["SELL", "SHORT"])].sort_values("alpha_score", ascending=True).head(10)
    qmd = _write_md_table(quantile_summary[["bucket", "number_of_stocks", "average_alpha", "long_selected_count", "short_selected_count"]]) if not quantile_summary.empty else "No quantile data."
    summary = [
        f"# Daily Factor Report - {signal_date.date().isoformat()}",
        "",
        "## Model Decision",
        f"Generated close-based signals for execution on {execution_date.date().isoformat() if execution_date is not None else 'the next trading day'}. Proposed orders: {len(orders)}. Expected turnover: {risk_summary.get('expected_turnover', 0):.4f}.",
        f"Orders require predicted alpha net of estimated trading cost to exceed {metadata.get('backtest_config', {}).get('min_net_alpha_after_cost_bps', 0):.2f} bps.",
        "",
        "## Buy Candidates",
        _write_md_table(buys[["ticker", "side", "target_weight", "alpha_score", "reason_text"]]) if not buys.empty else "No buy candidates.",
        "",
        "## Sell / Short Candidates",
        _write_md_table(sells[["ticker", "side", "target_weight", "alpha_score", "reason_text"]]) if not sells.empty else "No sell/short candidates.",
        "",
        "## Quantile Summary",
        qmd,
        "",
        "## Risk / Exposure",
        f"Gross: {risk_summary.get('gross_exposure', 0):.4f}; Net: {risk_summary.get('net_exposure', 0):.4f}; Long names: {risk_summary.get('number_long', 0)}; Short names: {risk_summary.get('number_short', 0)}.",
        "",
        "## Data Quality",
        f"Warnings: {', '.join(warnings) if warnings else 'none'}. Missing and finite factor coverage are recorded in pipeline_metadata.json.",
        "",
        "## Timing Convention",
        "Signals use data available at or before signal_date close. Orders are for the next trading day open. No intraday alpha recomputation is part of this workflow.",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return out


def run_after_close_signal_once(date: str | pd.Timestamp, cfg: DailyBacktestConfig) -> Path:
    signal_date = pd.Timestamp(date).normalize()
    _assert_signal_date_available(cfg, signal_date)
    static = _load_static_inputs(cfg)
    dates: pd.DatetimeIndex = static["dates"]
    tickers: list[str] = static["tickers"]
    factor_names: list[str] = static["factor_names"]
    signal_idx_arr = np.where(dates == signal_date)[0]
    if signal_idx_arr.size != 1:
        raise RuntimeError(f"signal_date={signal_date.date()} not uniquely present in dates.csv")
    t = int(signal_idx_arr[0])
    execution_date = dates[t + 1] if t + 1 < len(dates) else None

    input_dir = Path(cfg.input_dir)
    X = np.load(input_dir / "X.npy", mmap_mode="r")
    f = np.load(input_dir / "factor_returns.npy")
    r = np.load(input_dir / "r.npy", mmap_mode="r") if (input_dir / "r.npy").exists() else None
    tradable = np.load(input_dir / "tradable_mask.npy", mmap_mode="r")
    if tradable.shape != (len(dates), len(tickers)):
        raise ValueError("tradable_mask.npy shape does not match dates.csv and tickers.csv")
    if X.shape != (len(dates), len(tickers), len(factor_names)) or f.shape != (len(dates), len(factor_names)):
        raise ValueError("Factor arrays are not aligned to processed dates, securities, and factor names")
    universe = load_wrds_universe_mask(input_dir, dates, tickers).to_numpy(dtype=bool)
    tradable = np.asarray(tradable, dtype=bool) & universe
    for i, end in enumerate(static["delisting_dates"]):
        if pd.notna(end):
            tradable[dates >= end, i] = False
    market_bars_path = execution_bars_path(cfg)
    if market_bars_path is None:
        raise RuntimeError("After-close job requires daily_bars_path or execution_bars_path for price and liquidity filters.")
    f_pred = predict_factor_returns(
        f[: t + 1],
        method=cfg.forecast_method,
        lookback=cfg.lookback,
        ewma_halflife=cfg.ewma_halflife,
        min_periods=cfg.min_periods,
    )
    f_hat = f_pred[t]
    X_t = np.asarray(X[t], dtype=float)
    bars = _load_signal_bars(Path(market_bars_path), signal_date, tickers)
    close = pd.to_numeric(bars.get("raw_close", pd.Series(index=tickers, dtype=float)), errors="coerce").to_numpy(dtype=float)
    adjusted_close = pd.to_numeric(bars.get("close", pd.Series(index=tickers, dtype=float)), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(bars.get("volume", pd.Series(index=tickers, dtype=float)), errors="coerce").to_numpy(dtype=float)
    dollar_volume = adjusted_close * volume
    alpha = np.where(np.isfinite(X_t), X_t, 0.0) @ np.where(np.isfinite(f_hat), f_hat, 0.0)
    if not np.isfinite(f_hat).any():
        alpha[:] = np.nan
    target, filter_info = _rank_weights_one_day(
        alpha,
        np.asarray(tradable[t], dtype=bool),
        dollar_volume,
        cfg,
        price=close,
    )

    if t > 0:
        prev_hat = f_pred[t - 1] if t - 1 < len(f_pred) else np.full_like(f_hat, np.nan)
        prev_alpha = np.asarray(X[t - 1], dtype=float) @ np.where(np.isfinite(prev_hat), prev_hat, 0.0)
        prev_bars = _load_signal_bars(Path(market_bars_path), dates[t - 1], tickers)
        prev_dv = (
            pd.to_numeric(prev_bars.get("close", pd.Series(index=tickers, dtype=float)), errors="coerce").to_numpy(dtype=float)
            * pd.to_numeric(prev_bars.get("volume", pd.Series(index=tickers, dtype=float)), errors="coerce").to_numpy(dtype=float)
        )
        prev_close = pd.to_numeric(
            prev_bars.get("raw_close", pd.Series(index=tickers, dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=float)
        current, _ = _rank_weights_one_day(
            prev_alpha,
            np.asarray(tradable[t - 1], dtype=bool),
            prev_dv,
            cfg,
            price=prev_close,
        )
    else:
        current = np.zeros_like(target)
    if cfg.turnover_cap is not None:
        turnover = 0.5 * float(np.abs(target - current).sum())
        if turnover > cfg.turnover_cap and turnover > 0:
            target = current + (target - current) * (cfg.turnover_cap / turnover)
    target[~np.asarray(tradable[t], dtype=bool)] = 0.0
    if cfg.min_price > 0:
        price_ok = np.isfinite(close) & (close >= cfg.min_price)
        target[~price_ok] = 0.0

    alpha_rankings = _build_one_day_rankings(
        signal_date,
        tickers,
        static["company_names"],
        factor_names,
        X_t,
        f_hat,
        alpha,
        target,
        current,
        dollar_volume,
        close,
        cfg,
    )
    alpha_rankings["signal_price"] = close
    alpha_rankings["raw_signal_price"] = close
    alpha_rankings["symbol"] = alpha_rankings["ticker"].map(static["symbols"])
    alpha_rankings["source"] = "compustat_wrds"
    alpha_rankings["execution_date"] = execution_date
    alpha_rankings["execution_price"] = np.nan
    alpha_rankings["exit_date"] = pd.NaT
    alpha_rankings["exit_price"] = np.nan
    alpha_rankings["holding_period_days"] = cfg.holding_period_days
    target_positions = alpha_rankings[[
        "signal_date", "ticker", "symbol", "source", "target_weight", "current_weight", "order_weight_delta",
        "alpha_score", "trading_cost_bps", "net_alpha_after_cost_bps",
        "min_net_alpha_after_cost_bps", "passes_cost_threshold",
        "alpha_rank", "quantile_bucket", "price_flag", "reason_text",
    ]].copy()
    current_positions = target_positions.rename(columns={"current_weight": "weight"})[["signal_date", "ticker", "weight"]]
    factor_report = _build_factor_report(dates, t, factor_names, f, f_hat)
    residuals, outliers = (
        _build_residuals_latest(dates, tickers, X, r, f, t, cfg.residual_outlier_z_threshold)
        if r is not None
        else (pd.DataFrame(columns=["date", "ticker", "residual", "residual_zscore"]), pd.DataFrame())
    )

    quantile_rows = []
    finite_alpha = pd.Series(alpha).replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_alpha.empty:
        buckets = pd.qcut(finite_alpha, 10, labels=False, duplicates="drop") + 1 if len(finite_alpha) >= 10 else pd.Series(1, index=finite_alpha.index)
        for bucket in sorted(buckets.dropna().unique()):
            idx = buckets[buckets == bucket].index.to_numpy(dtype=int)
            quantile_rows.append({
                "signal_date": signal_date,
                "bucket": int(bucket),
                "alpha_min": float(np.nanmin(alpha[idx])),
                "alpha_max": float(np.nanmax(alpha[idx])),
                "number_of_stocks": int(len(idx)),
                "average_alpha": float(np.nanmean(alpha[idx])),
                "average_realized_next_day_return": np.nan,
                "average_residual": np.nan,
                "long_selected_count": int((target[idx] > 0).sum()),
                "short_selected_count": int((target[idx] < 0).sum()),
                "average_liquidity": float(np.nanmean(dollar_volume[idx])) if np.isfinite(dollar_volume[idx]).any() else np.nan,
                "average_volatility": np.nan,
                "long_threshold_alpha": filter_info.get("long_threshold_alpha"),
                "short_threshold_alpha": filter_info.get("short_threshold_alpha"),
                "filtered_out_price": filter_info.get("filtered_price"),
                "filtered_out_liquidity": filter_info.get("filtered_liquidity"),
                "filtered_out_missing_data": filter_info.get("filtered_missing_or_untradable"),
                "filtered_out_alpha_threshold": filter_info.get("filtered_alpha_threshold"),
                "cost_adjusted_alpha_threshold": filter_info.get("cost_adjusted_alpha_threshold"),
                "trading_cost_return": filter_info.get("trading_cost_return"),
                "min_net_alpha_after_cost": filter_info.get("min_net_alpha_after_cost"),
                "final_number_long": int((target > 0).sum()),
                "final_number_short": int((target < 0).sum()),
            })
    quantile_summary = pd.DataFrame(quantile_rows)

    orders = alpha_rankings[alpha_rankings["side"].ne("")].copy()
    execution_quality = pd.DataFrame({
        "date": execution_date,
        "ticker": orders["ticker"] if not orders.empty else pd.Series(dtype=str),
        "side": orders["side"] if not orders.empty else pd.Series(dtype=str),
        "planned_quantity": 0,
        "filled_quantity": 0,
        "fill_ratio": 0.0,
        "planned_price": orders["signal_price"] if not orders.empty else pd.Series(dtype=float),
        "average_fill_price": np.nan,
        "arrival_price": orders["signal_price"] if not orders.empty else pd.Series(dtype=float),
        "close_price": orders["signal_price"] if not orders.empty else pd.Series(dtype=float),
        "slippage_bps": cfg.slippage_bps,
        "transaction_cost": (orders["order_weight_delta"].abs() * cfg.initial_equity * cfg.transaction_cost_bps / 10_000.0) if not orders.empty else pd.Series(dtype=float),
        "estimated_slippage_cost": (orders["order_weight_delta"].abs() * cfg.initial_equity * cfg.slippage_bps / 10_000.0) if not orders.empty else pd.Series(dtype=float),
        "estimated_total_cost": (orders["order_weight_delta"].abs() * cfg.initial_equity * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0) if not orders.empty else pd.Series(dtype=float),
        "cost_basis": "order_notional",
        "order_status": "planned",
        "rejected_reason": "",
    })
    top_abs = pd.Series(target, index=tickers).abs().sort_values(ascending=False).head(10)
    risk_summary = {
        "gross_exposure": float(np.abs(target).sum()),
        "net_exposure": float(target.sum()),
        "long_exposure": float(np.clip(target, 0, None).sum()),
        "short_exposure": float(np.clip(target, None, 0).sum()),
        "number_long": int((target > 0).sum()),
        "number_short": int((target < 0).sum()),
        "max_position_weight": float(np.nanmax(np.abs(target))) if target.size else 0.0,
        "top_10_position_weights": {k: float(pd.Series(target, index=tickers).loc[k]) for k in top_abs.index},
        "sector_exposures": {},
        "beta_estimate": None,
        "expected_turnover": 0.5 * float(np.abs(target - current).sum()),
        "realized_turnover": None,
        "cash": None,
        "leverage": float(np.abs(target).sum()),
        "margin_usage": None,
        "liquidity_warnings": [],
    }
    finite_by_factor = np.isfinite(X_t).mean(axis=0)
    warnings = []
    bars_start, bars_end = daily_bars_date_range(cfg.daily_bars_path)
    exec_bars_start, exec_bars_end = daily_bars_date_range(market_bars_path)
    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _safe_git_hash(),
        "job": "after_close",
        "canonical_timing": "signal at close[t], execute next trading day open[t+1], no intraday signal recomputation",
        "data_start_date": dates[0].date().isoformat(),
        "data_end_date": dates[-1].date().isoformat(),
        "signal_date": signal_date.date().isoformat(),
        "execution_date": execution_date.date().isoformat() if execution_date is not None else None,
        "universe_size_raw": len(tickers),
        "universe_size_after_filters": int(np.isfinite(alpha).sum()),
        "factor_names": factor_names,
        "model_config": {"forecast_method": cfg.forecast_method, "lookback": cfg.lookback, "ewma_halflife": cfg.ewma_halflife},
        "backtest_config": {
            **cfg.__dict__,
            "input_dir": str(cfg.input_dir),
            "out_dir": str(cfg.out_dir),
            "daily_bars_path": str(cfg.daily_bars_path),
            "execution_bars_path": str(market_bars_path),
            "reports_dir": str(cfg.reports_dir),
        },
        "data_source": "compustat_wrds",
        "universe": "historical_nasdaq_exchg_14",
        "exclude_bottom_market_cap_fraction": 0.20,
        "input_files": {
            "input_dir": str(cfg.input_dir),
            "daily_bars_path": str(cfg.daily_bars_path),
            "execution_bars_path": str(market_bars_path),
            "processed_dates_start": dates[0].date().isoformat(),
            "processed_dates_end": dates[-1].date().isoformat(),
            "daily_bars_start": bars_start,
            "daily_bars_end": bars_end,
            "execution_bars_start": exec_bars_start,
            "execution_bars_end": exec_bars_end,
        },
        "warnings": warnings,
        "data_quality": {
            "missing_factor_coverage": {name: float(1.0 - finite_by_factor[i]) for i, name in enumerate(factor_names)},
            "nan_counts_by_factor": {name: int(np.isnan(X_t[:, i]).sum()) for i, name in enumerate(factor_names)},
            "finite_fraction_by_factor": {name: float(finite_by_factor[i]) for i, name in enumerate(factor_names)},
            "excluded_tickers_latest": int((~np.asarray(tradable[t], dtype=bool)).sum()),
        },
    }
    return _write_single_date_report(
        Path(cfg.reports_dir),
        signal_date,
        execution_date,
        alpha_rankings,
        target_positions,
        current_positions,
        factor_report,
        quantile_summary,
        residuals,
        outliers,
        execution_quality,
        risk_summary,
        metadata,
        warnings,
    )


def run_after_close_job(date: str | pd.Timestamp, config_path: str | Path | None = None) -> Path:
    """Create close-based signals and a next-day order plan.

    This job intentionally computes alpha once, after the selected signal date's
    close, and writes artifacts for next-open execution.
    """
    cfg = load_daily_job_config(config_path)
    report_dir = run_after_close_signal_once(date, cfg)
    order_plan = report_dir / "orders.csv"
    if not order_plan.exists():
        raise RuntimeError(f"after-close job did not create order plan: {order_plan}")
    return report_dir


def run_next_open_execution_job(date: str | pd.Timestamp, config_path: str | Path | None = None) -> Path:
    """Load the prior signal report and mark simulated next-open execution.

    Live broker integration can replace the fill section here; the important
    convention is that this job reads an existing order plan and does not
    recompute alpha from intraday bars.
    """
    cfg = load_daily_job_config(config_path)
    execution_date = pd.Timestamp(date).normalize()
    candidates = sorted(Path(cfg.reports_dir).glob("*/orders.csv"))
    if not candidates:
        raise FileNotFoundError(f"No after-close order plans found under {cfg.reports_dir}")
    order_path = candidates[-1]
    report_dir = order_path.parent
    orders = pd.read_csv(order_path)
    eq_path = report_dir / "execution_quality.csv"
    if eq_path.exists():
        execution = pd.read_csv(eq_path)
    else:
        execution = pd.DataFrame()
    if not execution.empty:
        execution["date"] = execution_date.date().isoformat()
        execution["order_status"] = execution.get("order_status", "filled")
        execution.to_csv(eq_path, index=False)
    metadata_path = report_dir / "pipeline_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["next_open_execution_job"] = {
        "execution_date": execution_date.date().isoformat(),
        "loaded_order_plan": str(order_path),
        "orders_loaded": int(len(orders)),
        "note": "No alpha recomputation performed in next-open execution job.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report_dir
