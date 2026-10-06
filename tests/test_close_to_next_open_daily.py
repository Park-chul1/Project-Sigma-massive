from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtests.close_to_next_open_daily import (
    DailyBacktestConfig,
    REQUIRED_REPORT_FILES,
    apply_dynamic_risk_controls,
    _rank_weights_one_day,
    compute_execution_returns,
    compute_alpha_and_contributions,
    portfolio_pnl,
    run_close_to_next_open_backtest,
    write_daily_report,
)


def _write_fixture(tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=6)
    tickers = [f"T{i:02d}" for i in range(10)]
    factors = ["momentum", "quality", "liquidity"]
    X = np.zeros((len(dates), len(tickers), len(factors)), dtype=float)
    for t in range(len(dates)):
        X[t, :, 0] = np.linspace(-1, 1, len(tickers))
        X[t, :, 1] = np.linspace(1, -1, len(tickers)) * 0.2
        X[t, :, 2] = 0.1
    f = np.tile(np.array([0.02, -0.01, 0.005]), (len(dates), 1))
    f[0] = np.nan
    r = np.tile(np.linspace(-0.01, 0.01, len(tickers)), (len(dates), 1))
    tradable = np.ones((len(dates), len(tickers)), dtype=bool)
    np.save(input_dir / "X.npy", X)
    np.save(input_dir / "factor_returns.npy", f)
    np.save(input_dir / "r.npy", r)
    np.save(input_dir / "tradable_mask.npy", tradable)
    pd.DataFrame({"date": dates}).to_csv(input_dir / "dates.csv", index=False)
    pd.DataFrame({"ticker": tickers, "source": "compustat_wrds", "list_date": dates[0], "delisted_utc": pd.NaT}).to_csv(input_dir / "tickers.csv", index=False)
    pd.DataFrame({"ticker": tickers, "from_date": dates[0], "thru_date": pd.NaT, "source": "compustat_wrds"}).to_parquet(input_dir / "nasdaq_membership.parquet", index=False)
    pd.DataFrame({"factor": factors}).to_csv(input_dir / "factor_names.csv", index=False)

    rows = []
    for t, date in enumerate(dates):
        for i, ticker in enumerate(tickers):
            price = 20 + i + t
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": price,
                "close": price + 0.5,
                "volume": 1_000_000 + i,
                "market_cap": (abs(i - 4.5) + 1) * 1_000_000_000,
                "source": "compustat_wrds",
            })
    bars_path = tmp_path / "bars.parquet"
    pd.DataFrame(rows).to_parquet(bars_path, index=False)
    return input_dir, bars_path, dates, tickers, factors, X, f


def _cfg(tmp_path) -> DailyBacktestConfig:
    input_dir, bars_path, *_ = _write_fixture(tmp_path)
    return DailyBacktestConfig(
        input_dir=input_dir,
        out_dir=tmp_path / "bt",
        daily_bars_path=bars_path,
        reports_dir=tmp_path / "reports",
        min_names_per_side=1,
        max_position_weight=1.0,
        gross_exposure=2.0,
        forecast_method="latest",
        min_periods=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )


def test_timing_and_execution_uses_next_open(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    daily = result.daily.dropna(subset=["execution_date"])
    assert (pd.to_datetime(daily["signal_date"]) < pd.to_datetime(daily["execution_date"])).all()
    first_order = result.orders.dropna(subset=["execution_price"]).iloc[0]
    bars = pd.read_parquet(cfg.daily_bars_path)
    expected_open = bars[(pd.to_datetime(bars["date"]).dt.normalize() == pd.Timestamp(first_order["execution_date"]).normalize()) & (bars["ticker"] == first_order["ticker"])]["open"].iloc[0]
    assert first_order["execution_price"] == pytest.approx(expected_open)


def test_oracle_forecast_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    bad = cfg.__class__(**{**cfg.__dict__, "forecast_method": "oracle"})
    with pytest.raises(ValueError):
        run_close_to_next_open_backtest(bad)


def test_portfolio_quantiles_and_exposure(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    ranked = result.alpha_rankings[pd.to_datetime(result.alpha_rankings["signal_date"]).dt.normalize().eq(pd.Timestamp("2024-01-04"))]
    longs = ranked[ranked["target_weight"] > 0]
    shorts = ranked[ranked["target_weight"] < 0]
    assert not longs.empty
    assert not shorts.empty
    assert longs["alpha_score"].min() >= ranked["alpha_score"].quantile(cfg.long_quantile) - 1e-12
    assert shorts["alpha_score"].max() <= ranked["alpha_score"].quantile(cfg.short_quantile) + 1e-12
    latest = result.daily[pd.to_datetime(result.daily["signal_date"]).dt.normalize().eq(pd.Timestamp("2024-01-04"))].iloc[0]
    assert latest["gross_exposure"] == pytest.approx(2.0)
    assert latest["net_exposure"] == pytest.approx(0.0)


def test_cost_adjusted_alpha_threshold_blocks_weak_signals(tmp_path):
    cfg = DailyBacktestConfig(
        min_names_per_side=1,
        max_position_weight=1.0,
        gross_exposure=2.0,
        transaction_cost_bps=1.0,
        slippage_bps=2.0,
        min_net_alpha_after_cost_bps=5.0,
    )
    alpha = np.array([0.0007, 0.0009, -0.0007, -0.0009])
    target, info = _rank_weights_one_day(alpha, np.ones_like(alpha, dtype=bool), np.full_like(alpha, 1_000_000.0), cfg)
    assert target[1] > 0
    assert target[3] < 0
    assert target[0] == 0
    assert target[2] == 0
    assert info["cost_adjusted_alpha_threshold"] == pytest.approx(0.0008)
    assert info["filtered_alpha_threshold"] == 2


def test_min_price_filter_blocks_low_price_names():
    cfg = DailyBacktestConfig(
        min_names_per_side=1,
        min_price=5.0,
        max_position_weight=1.0,
        gross_exposure=2.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        min_net_alpha_after_cost_bps=0.0,
    )
    alpha = np.array([-0.04, -0.03, 0.03, 0.04])
    price = np.array([4.99, 5.00, 4.50, 6.00])

    target, info = _rank_weights_one_day(
        alpha,
        np.ones_like(alpha, dtype=bool),
        np.full_like(alpha, 1_000_000.0),
        cfg,
        price=price,
    )

    assert target[0] == 0
    assert target[2] == 0
    assert target[1] < 0
    assert target[3] > 0
    assert info["filtered_price"] == 2


def test_net_exposure_limit_reduces_long_only_book():
    cfg = DailyBacktestConfig(
        min_names_per_side=1,
        allow_short=False,
        max_position_weight=1.0,
        gross_exposure=1.0,
        max_net_exposure=0.20,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    alpha = np.array([0.01, 0.02, 0.03, 0.04])
    target, _info = _rank_weights_one_day(alpha, np.ones_like(alpha, dtype=bool), None, cfg)
    assert abs(target.sum()) <= cfg.max_net_exposure + 1e-12
    assert np.abs(target).max() <= cfg.max_position_weight + 1e-12


def test_dynamic_risk_controls_scale_down_on_high_var():
    weights = np.tile(np.array([0.5, -0.5]), (5, 1))
    realized = np.array([
        [0.10, -0.10],
        [-0.10, 0.10],
        [0.10, -0.10],
        [-0.10, 0.10],
        [0.10, -0.10],
    ])
    cfg = DailyBacktestConfig(
        max_daily_var=0.02,
        var_lookback_days=2,
        var_confidence=0.95,
        volatility_target_annual=None,
        max_drawdown_limit=None,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    adjusted, controls = apply_dynamic_risk_controls(weights, realized, cfg)
    assert controls["risk_scale"].iloc[3] < 1.0
    assert np.abs(adjusted[3]).sum() < np.abs(weights[3]).sum()


def test_turnover_penalty_is_deducted_from_returns():
    weights = np.array([[0.5, -0.5], [0.0, 0.0]])
    realized = np.zeros_like(weights)
    cfg = DailyBacktestConfig(transaction_cost_bps=0.0, slippage_bps=0.0, turnover_penalty_bps=10.0)
    returns, turnover, costs, penalty = portfolio_pnl(weights, realized, cfg)
    assert turnover[0] == pytest.approx(0.5)
    assert costs[0] == 0
    assert penalty[0] == pytest.approx(0.001)
    assert returns[0] == pytest.approx(-0.001)


def test_transaction_cost_uses_full_traded_notional():
    weights = np.array([[0.5, -0.5], [0.0, 0.0]])
    realized = np.zeros_like(weights)
    cfg = DailyBacktestConfig(transaction_cost_bps=10.0, slippage_bps=0.0, turnover_penalty_bps=0.0)
    returns, turnover, costs, penalty = portfolio_pnl(weights, realized, cfg)
    assert turnover[0] == pytest.approx(0.5)
    assert costs[0] == pytest.approx(0.001)
    assert penalty[0] == 0
    assert returns[0] == pytest.approx(-0.001)


def test_execution_bars_path_drives_min_price_filter(tmp_path):
    input_dir, bars_path, _dates, _tickers, *_ = _write_fixture(tmp_path)
    raw = pd.read_parquet(bars_path)
    raw.loc[raw["ticker"].eq("T09"), ["open", "close"]] = 4.0
    raw_path = tmp_path / "raw_bars.parquet"
    raw.to_parquet(raw_path, index=False)
    cfg = DailyBacktestConfig(
        input_dir=input_dir,
        out_dir=tmp_path / "bt",
        daily_bars_path=bars_path,
        execution_bars_path=raw_path,
        reports_dir=tmp_path / "reports",
        min_names_per_side=1,
        max_positions_long=1,
        max_positions_short=1,
        max_position_weight=1.0,
        gross_exposure=2.0,
        forecast_method="latest",
        min_periods=1,
        min_price=5.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        min_net_alpha_after_cost_bps=0.0,
    )
    result = run_close_to_next_open_backtest(cfg)
    t09 = result.target_positions[result.target_positions["ticker"].eq("T09")]
    assert t09["target_weight"].max() == 0


def test_missing_exit_price_raises_for_active_position(tmp_path):
    cfg = _cfg(tmp_path)
    membership_path = cfg.input_dir / "nasdaq_membership.parquet"
    membership = pd.read_parquet(membership_path)
    membership.loc[membership["ticker"].eq("T09"), "thru_date"] = pd.Timestamp("2024-01-04")
    membership.to_parquet(membership_path, index=False)
    bars = pd.read_parquet(cfg.daily_bars_path)
    dates = sorted(pd.to_datetime(bars["date"]).dt.normalize().unique())
    bars.loc[
        pd.to_datetime(bars["date"]).dt.normalize().eq(pd.Timestamp(dates[4])) & bars["ticker"].eq("T09"),
        "open",
    ] = np.nan
    bars.to_parquet(cfg.daily_bars_path, index=False)
    with pytest.raises(RuntimeError, match="Missing exit price"):
        run_close_to_next_open_backtest(cfg)


def test_reports_are_generated_and_reasons_present(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    report_dir = write_daily_report(result, cfg.reports_dir, signal_date=pd.Timestamp("2024-01-03"))
    for name in REQUIRED_REPORT_FILES:
        assert (report_dir / name).exists(), name
    assert (report_dir / "factor_dashboard.png").exists()
    orders = pd.read_csv(report_dir / "orders.csv")
    assert orders["reason_text"].fillna("").str.len().gt(0).all()
    metadata = json.loads((report_dir / "pipeline_metadata.json").read_text())
    assert metadata["signal_date"] < metadata["execution_date"]


def test_factor_contributions_sum_to_alpha(tmp_path):
    input_dir, _bars_path, _dates, _tickers, _factors, X, f = _write_fixture(tmp_path)
    f_pred = np.vstack([np.full(3, np.nan), f[:-1]])
    alpha, contrib = compute_alpha_and_contributions(X, f_pred)
    mask = np.isfinite(alpha)
    assert np.allclose(contrib.sum(axis=2)[mask], alpha[mask])


def test_residual_heatmap_handles_many_tickers(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    report_dir = write_daily_report(result, cfg.reports_dir, signal_date=pd.Timestamp("2024-01-03"))
    assert (report_dir / "residual_heatmap.png").stat().st_size > 0


def test_kalman_diagnostic_columns_are_reported(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_close_to_next_open_backtest(cfg)
    cols = set(result.factor_forecast.columns)
    assert {"kalman_f_hat", "kalman_gain", "state_covariance_diag", "process_noise_q", "measurement_noise_r", "final_f_hat"}.issubset(cols)


def test_legacy_source_and_missing_cap_are_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    bars = pd.read_parquet(cfg.daily_bars_path)
    bars.drop(columns=["market_cap"]).to_parquet(cfg.daily_bars_path, index=False)
    with pytest.raises(ValueError, match="market_cap"):
        run_close_to_next_open_backtest(cfg)
    bars["source"] = "legacy"
    bars.to_parquet(cfg.daily_bars_path, index=False)
    with pytest.raises(ValueError, match="Compustat/WRDS"):
        run_close_to_next_open_backtest(cfg)


def test_membership_and_cap_exclusions_override_turnover_in_both_layers(tmp_path):
    cfg = _cfg(tmp_path)
    cfg = cfg.__class__(**{**cfg.__dict__, "turnover_cap": 0.1})
    bars = pd.read_parquet(cfg.daily_bars_path)
    # T09 becomes the smallest security after it has already been held.
    bars.loc[(bars["date"] >= pd.Timestamp("2024-01-05")) & bars["ticker"].eq("T09"), "market_cap"] = 1.0
    bars.to_parquet(cfg.daily_bars_path, index=False)
    membership = pd.read_parquet(cfg.input_dir / "nasdaq_membership.parquet")
    membership.loc[membership["ticker"].eq("T00"), "thru_date"] = pd.Timestamp("2024-01-04")
    membership.to_parquet(cfg.input_dir / "nasdaq_membership.parquet", index=False)
    result = run_close_to_next_open_backtest(cfg)
    targets = result.target_positions
    for ticker in ["T00", "T09"]:
        selected = targets[targets["ticker"].eq(ticker)]
        assert selected.loc[selected["signal_date"] < "2024-01-05", "target_weight"].abs().max() > 0
        assert selected.loc[selected["signal_date"] >= "2024-01-05", "target_weight"].eq(0).all()


def _delisting_fixture():
    dates = pd.bdate_range("2024-01-02", periods=5)
    # Recovery is based on the latest 80 close, not the 100 entry or stale 999.
    panels = {
        "open": pd.DataFrame({"A": [90.0, 100.0, 110.0, 999.0, 999.0]}, index=dates),
        "close": pd.DataFrame({"A": [95.0, 120.0, 80.0, 999.0, 999.0]}, index=dates),
    }
    meta = pd.DataFrame({"ticker": ["A"], "delisted_utc": [dates[3]]})
    return dates, panels, meta


@pytest.mark.parametrize("weight,expected_pnl", [(1.0, -0.6), (-1.0, 0.6)])
def test_confirmed_delisting_recovers_half_last_price_for_long_and_short(weight, expected_pnl):
    dates, panels, meta = _delisting_fixture()
    weights = np.zeros((len(dates), 1))
    weights[0, 0] = weight
    cfg = DailyBacktestConfig(holding_period_days=2, transaction_cost_bps=0, slippage_bps=0, max_abs_return=0.1)
    realized, prices, warnings = compute_execution_returns(weights, dates, panels, None, cfg, ticker_metadata=meta)
    assert realized[0, 0] == pytest.approx(-0.6)
    assert prices.iloc[0]["exit_price"] == pytest.approx(40.0)
    assert prices.iloc[0]["exit_date"] == dates[3]
    assert any("50%" in warning for warning in warnings)
    pnl, *_ = portfolio_pnl(weights, realized, cfg)
    assert pnl[0] == pytest.approx(expected_pnl)


def test_delisting_on_entry_prevents_purchase_even_with_stale_quote():
    dates, panels, meta = _delisting_fixture()
    meta["delisted_utc"] = dates[1]
    weights = np.zeros((len(dates), 1))
    weights[0, 0] = 1.0
    realized, prices, _ = compute_execution_returns(weights, dates, panels, None, DailyBacktestConfig(), ticker_metadata=meta)
    assert weights[0, 0] == 0
    assert np.isnan(realized[0, 0])
    assert np.isnan(prices.iloc[0]["execution_price"])


def test_delisting_uses_adjusted_price_basis_and_falls_back_to_last_valid_quote():
    dates, panels, meta = _delisting_fixture()
    # No closing quote on the final trading day: its open is the last price.
    panels["open"].iloc[2, 0] = 70.0
    panels["close"].iloc[2, 0] = np.nan
    panels["raw_open"] = panels["open"] * 2.0
    panels["raw_close"] = panels["close"] * 2.0
    panels["raw_open"].iloc[1, 0] = 1000.0  # Split basis differs at entry.
    weights = np.zeros((len(dates), 1))
    weights[0, 0] = 1.0
    realized, prices, _ = compute_execution_returns(weights, dates, panels, None, DailyBacktestConfig(holding_period_days=2), ticker_metadata=meta)
    assert realized[0, 0] == pytest.approx(35.0 / 100.0 - 1.0)
    assert prices.iloc[0]["execution_price"] == 1000.0
    assert prices.iloc[0]["exit_price"] == 70.0


def test_an_ordinary_quote_gap_is_not_a_delisting():
    dates, panels, _ = _delisting_fixture()
    panels["open"].iloc[2, 0] = np.nan
    weights = np.zeros((len(dates), 1))
    weights[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="Missing exit price"):
        compute_execution_returns(weights, dates, panels, None, DailyBacktestConfig())


def test_zero_or_legacy_terminal_return_policy_cannot_be_enabled():
    for policy in ["zero", "terminal_return"]:
        with pytest.raises(ValueError, match="confirmed delistings"):
            DailyBacktestConfig(missing_price_policy=policy).validate()
    with pytest.raises(ValueError, match="mandatory"):
        DailyBacktestConfig(use_point_in_time_universe=False).validate()
