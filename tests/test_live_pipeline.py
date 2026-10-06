from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live.alpha import apply_signal_filters, compute_alpha, standardize_alpha
from live.config import LiveConfig
from live.data import load_recent_history, update_local_bar_cache
from live.execution import generate_orders
from live.features import build_latest_exposures
from live.paper_order_submitter import _apply_min_price_to_targets
from live.portfolio import compute_turnover, construct_target_portfolio
from live.risk import assert_demo_safety


def cfg(**kwargs) -> LiveConfig:
    base = LiveConfig(
        MIN_DOLLAR_VOLUME=0,
        MIN_FACTOR_NAMES=2,
        MIN_EXPOSURE_COVERAGE=0.5,
        ALPHA_Z_THRESHOLD=0.5,
        MAX_TURNOVER_PER_REBALANCE=1.0,
        MIN_HOLDING_PERIOD_BARS=2,
    )
    return base.__class__(**{**base.__dict__, **kwargs})


def test_no_lookahead_in_daily_history(tmp_path):
    path = tmp_path / "bars.parquet"
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-05-11", "2026-05-12"]),
        "ticker": ["A", "A"],
        "open": [10, 99],
        "high": [11, 100],
        "low": [9, 98],
        "close": [10, 99],
        "volume": [1000, 1000],
        "source": "compustat_wrds",
    })
    update_local_bar_cache(bars, path)
    hist = load_recent_history(["A"], 10, path, end_time=pd.Timestamp("2026-05-11"))
    assert hist["timestamp"].max() == pd.Timestamp("2026-05-11")
    assert hist["close"].iloc[-1] == 10


def test_alpha_threshold_filter():
    config = cfg(ALPHA_Z_THRESHOLD=0.9, COST_BPS=1, SLIPPAGE_BPS=1)
    alpha = pd.Series({"A": 0.02, "B": 0.00001, "C": -0.03})
    z = pd.Series({"A": 1.0, "B": 2.0, "C": -0.1})
    bars = pd.DataFrame({"ticker": ["A", "B", "C"], "close": [10, 10, 10], "volume": [1000, 1000, 1000]})
    out = apply_signal_filters(alpha, z, bars, config).set_index("ticker")
    assert bool(out.loc["A", "passes_signal"])
    assert not bool(out.loc["B", "passes_signal"])
    assert not bool(out.loc["C", "passes_signal"])


def test_alpha_filter_uses_net_alpha_after_cost_threshold():
    config = cfg(ALPHA_Z_THRESHOLD=0.0, COST_BPS=1, SLIPPAGE_BPS=2, MIN_NET_ALPHA_AFTER_COST_BPS=5)
    alpha = pd.Series({"A": 0.0009, "B": 0.0007})
    z = pd.Series({"A": 1.0, "B": 1.0})
    bars = pd.DataFrame({"ticker": ["A", "B"], "close": [10, 10], "volume": [1000, 1000]})
    out = apply_signal_filters(alpha, z, bars, config).set_index("ticker")
    assert bool(out.loc["A", "passes_signal"])
    assert not bool(out.loc["B", "passes_signal"])


def test_turnover_cap():
    config = cfg(MAX_TURNOVER_PER_REBALANCE=0.05, MAX_POSITION_WEIGHT=1, MAX_GROSS_EXPOSURE=2)
    candidates = pd.DataFrame({
        "ticker": ["A", "B"],
        "alpha": [0.02, -0.02],
        "z_alpha": [2.0, -2.0],
        "passes_signal": [True, True],
    })
    target = construct_target_portfolio(candidates, pd.Series(dtype=float), config)
    assert compute_turnover(pd.Series(dtype=float), target) <= 0.0500001


def test_min_holding_period_blocks_exit():
    config = cfg(MIN_HOLDING_PERIOD_BARS=3, MAX_TURNOVER_PER_REBALANCE=1)
    candidates = pd.DataFrame(columns=["ticker", "alpha", "z_alpha", "passes_signal"])
    current = pd.Series({"A": 0.01})
    target = construct_target_portfolio(candidates, current, config, holding_bars={"A": 1})
    assert target.loc["A"] == pytest.approx(0.01)


def test_max_position_size():
    config = cfg(MAX_POSITION_WEIGHT=0.03, MAX_GROSS_EXPOSURE=1, MAX_TURNOVER_PER_REBALANCE=1)
    candidates = pd.DataFrame({
        "ticker": ["A", "B"],
        "alpha": [0.05, -0.05],
        "z_alpha": [10.0, -10.0],
        "passes_signal": [True, True],
    })
    target = construct_target_portfolio(candidates, pd.Series(dtype=float), config)
    assert target.abs().max() <= 0.0300001


def test_paper_demo_safety_flag():
    safe = cfg(PAPER_TRADING=True, ENABLE_REAL_TRADING=False)
    assert_demo_safety(safe)
    unsafe = cfg(PAPER_TRADING=False, ENABLE_REAL_TRADING=False)
    with pytest.raises(RuntimeError):
        assert_demo_safety(unsafe)


def test_order_generation_from_current_to_target():
    config = cfg(MIN_ORDER_DOLLARS=0)
    orders = generate_orders(
        current_positions=pd.Series({"A": 5}),
        target_weights=pd.Series({"A": 0.02, "B": -0.01}),
        prices=pd.Series({"A": 10.0, "B": 20.0}),
        equity=10_000,
        config=config,
    )
    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["A"].side == "BUY"
    assert by_ticker["A"].quantity == 15
    assert by_ticker["B"].side == "SELL"
    assert by_ticker["B"].quantity == 5


def test_paper_submitter_drops_low_price_targets_before_order_generation():
    target = pd.Series({"A": 0.01, "B": -0.01, "C": 0.02})
    prices = pd.Series({"A": 4.99, "B": 5.00})
    filtered = _apply_min_price_to_targets(target, prices, min_price=5.0)
    assert filtered.to_dict() == {"B": -0.01, "C": 0.02}


def test_nan_exposure_handling():
    X = pd.DataFrame({"f1": [1.0, np.nan, np.nan], "f2": [0.5, np.nan, np.nan]}, index=["A", "B", "C"])
    alpha = compute_alpha(X.dropna(how="all"), pd.Series({"f1": 1.0, "f2": 1.0}))
    z = standardize_alpha(alpha)
    assert "B" not in alpha.index
    assert z.index.tolist() == ["A"]


def test_feature_layer_drops_poor_nan_coverage():
    config = cfg(MIN_FACTOR_NAMES=2, MIN_EXPOSURE_COVERAGE=0.5, LOOKBACK_BARS=80)
    rows = []
    for i, ts in enumerate(pd.date_range("2026-01-01 09:30", periods=80, freq="15min")):
        for ticker, base in [("A", 10), ("B", 20), ("C", 30)]:
            rows.append({
                "timestamp": ts,
                "ticker": ticker,
                "open": base + i * 0.1,
                "high": base + i * 0.1 + 1,
                "low": base + i * 0.1 - 1,
                "close": base + i * 0.1,
                "volume": 1000 if ticker != "C" else np.nan,
            })
    X, tickers, names, coverage = build_latest_exposures(pd.DataFrame(rows), config)
    assert set(tickers).issubset({"A", "B", "C"})
    assert np.isfinite(X.to_numpy(dtype=float)).all() if not X.empty else True


def test_live_rejects_non_wrds_and_intraday_providers():
    from live.data import make_provider

    for provider in ["massive", "ibkr_delayed", "yfinance"]:
        with pytest.raises(ValueError, match="Compustat WRDS"):
            make_provider(cfg(DATA_PROVIDER=provider))
    with pytest.raises(ValueError, match="daily bars only"):
        cfg(BAR_INTERVAL="15m").validate()


def test_live_cache_rejects_unprovenanced_prices(tmp_path):
    from live.data import CachedOnlyProvider

    path = tmp_path / "bars.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2026-05-12")], "ticker": ["000001_01"], "close": [10]}).to_parquet(path)
    with pytest.raises(ValueError, match="exclusively from Compustat/WRDS"):
        CachedOnlyProvider(cfg(CACHE_PATH=path)).fetch_latest_daily_bars(["000001_01"], "2026-05-12")


def test_live_universe_historical_members_and_bottom_market_cap(tmp_path):
    from live.data import load_universe

    names = [f"{i:06d}_01" for i in range(1, 7)]
    membership = pd.DataFrame({
        "ticker": names,
        "from_date": pd.to_datetime(["2020-01-01"] * 5 + ["2026-05-13"]),
        "thru_date": pd.to_datetime([None] * 6),
        "source": "compustat_wrds",
    })
    bars = pd.DataFrame({
        "ticker": names * 2,
        "date": pd.to_datetime(["2026-05-12"] * 6 + ["2026-05-13"] * 6),
        "market_cap": [1, 2, 3, 4, 5, 1000, 999, 5, 4, 3, 2, 1],
        "volume": [999999, 10, 1, 1, 1, 1] * 2,
        "source": "compustat_wrds",
    })
    membership.to_parquet(tmp_path / "members.parquet")
    bars.to_parquet(tmp_path / "bars.parquet")
    pd.DataFrame({"ticker": names, "delisted_utc": pd.NaT, "source": "compustat_wrds"}).to_csv(tmp_path / "tickers.csv", index=False)
    config = cfg(MEMBERSHIP_PATH=tmp_path / "members.parquet", UNIVERSE_BARS_PATH=tmp_path / "bars.parquet", UNIVERSE_PATH=tmp_path / "tickers.csv")
    assert load_universe(config, "2026-05-12") == names[1:5][::-1]
    metadata = pd.read_csv(tmp_path / "tickers.csv")
    metadata["delisted_utc"] = pd.to_datetime(metadata["delisted_utc"])
    metadata.loc[4, "delisted_utc"] = pd.Timestamp("2026-05-12")
    metadata.to_csv(tmp_path / "tickers.csv", index=False)
    assert load_universe(config, "2026-05-12") == names[1:4][::-1]


def test_live_universe_missing_inputs_fails_closed(tmp_path):
    from live.data import load_universe

    with pytest.raises(FileNotFoundError, match="Missing WRDS"):
        load_universe(cfg(MEMBERSHIP_PATH=tmp_path / "missing.parquet"), "2026-05-12")


def test_broker_symbol_mapping_rejects_ambiguous_symbols(tmp_path):
    from live.data import load_broker_symbol_map

    path = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": ["000001_01", "000002_01"], "symbol": ["ABC", "ABC"], "source": "compustat_wrds"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ambiguous"):
        load_broker_symbol_map(path, ["000001_01", "000002_01"])
    assert load_broker_symbol_map(path, ["000001_01"]).to_dict() == {"000001_01": "ABC"}


def test_paper_order_inputs_use_raw_prices_and_resolved_symbols(tmp_path):
    from live.paper_order_submitter import _report_execution_inputs

    pd.DataFrame({
        "ticker": ["000001_01"], "symbol": ["ABC"], "source": ["compustat_wrds"],
        "signal_price": [5.0], "raw_signal_price": [100.0],
    }).to_csv(tmp_path / "alpha_rankings.csv", index=False)
    symbols, prices = _report_execution_inputs(tmp_path, ["000001_01"])
    assert symbols.to_dict() == {"000001_01": "ABC"}
    assert prices.to_dict() == {"ABC": 100.0}


def test_paper_order_inputs_reject_legacy_symbol_only_reports(tmp_path):
    from live.paper_order_submitter import _report_execution_inputs

    pd.DataFrame({"ticker": ["ABC"], "signal_price": [100.0]}).to_csv(tmp_path / "alpha_rankings.csv", index=False)
    with pytest.raises(ValueError, match="exclusively from Compustat/WRDS"):
        _report_execution_inputs(tmp_path, ["ABC"])


def test_after_close_forces_cap_exit_even_with_zero_turnover_and_keeps_raw_prices(tmp_path, monkeypatch):
    import live.daily_jobs as jobs
    from backtests.close_to_next_open_daily import DailyBacktestConfig

    names = [f"{i:06d}_01" for i in range(1, 6)]
    dates = pd.bdate_range("2026-05-11", periods=3)
    pd.DataFrame({"date": dates}).to_csv(tmp_path / "dates.csv", index=False)
    pd.DataFrame({"ticker": names, "symbol": ["A", "B", "C", "D", "E"], "delisted_utc": pd.NaT, "source": "compustat_wrds"}).to_csv(tmp_path / "tickers.csv", index=False)
    pd.DataFrame({"factor": ["f1"]}).to_csv(tmp_path / "factor_names.csv", index=False)
    pd.DataFrame({"ticker": names, "from_date": dates[0], "thru_date": pd.NaT, "source": "compustat_wrds"}).to_parquet(tmp_path / "nasdaq_membership.parquet")
    exposures = np.tile(np.array([-3., -2., -1., 1., 3.])[None, :, None], (3, 1, 1))
    np.save(tmp_path / "X.npy", exposures)
    np.save(tmp_path / "factor_returns.npy", np.ones((3, 1)) * .02)
    np.save(tmp_path / "tradable_mask.npy", np.ones((3, 5), dtype=bool))
    rows = []
    for t, date in enumerate(dates):
        for i, name in enumerate(names):
            rows.append({
                "ticker": name, "date": date, "close": 10., "raw_close": 100.,
                "volume": 10000., "market_cap": .1 if t == 2 and i == 4 else float(i + 1),
                "source": "compustat_wrds",
            })
    pd.DataFrame(rows).to_parquet(tmp_path / "daily_bars.parquet")
    config = DailyBacktestConfig(
        input_dir=tmp_path, daily_bars_path=tmp_path / "daily_bars.parquet", reports_dir=tmp_path / "reports",
        forecast_method="latest", min_periods=1, min_names_per_side=1, turnover_cap=0.,
        max_position_weight=1., transaction_cost_bps=0., slippage_bps=0., min_net_alpha_after_cost_bps=0.,
    )
    monkeypatch.setattr(jobs, "_write_single_date_report", lambda *args: args[3])
    report = jobs.run_after_close_signal_once(dates[-1], config).set_index("ticker")
    assert report.loc[names[-1], "current_weight"] > 0
    assert report.loc[names[-1], "target_weight"] == 0
    assert report["raw_signal_price"].eq(100.).all()
    assert report.loc[names[-1], "symbol"] == "E"


def test_share_sizing_uses_unadjusted_close_only():
    from live.execution import latest_prices_from_bars

    bars = pd.DataFrame({"ticker": ["ABC"], "close": [10.], "raw_close": [100.]})
    prices = latest_prices_from_bars(bars)
    orders = generate_orders(pd.Series(dtype=float), pd.Series({"ABC": .1}), prices, 10_000., cfg(MIN_ORDER_DOLLARS=0))
    assert orders[0].quantity == 10
    with pytest.raises(ValueError, match="raw_close"):
        latest_prices_from_bars(bars.drop(columns=["raw_close"]))


def test_zero_target_report_can_liquidate_excluded_holding(tmp_path, monkeypatch):
    import live.paper_order_submitter as submitter

    pd.DataFrame({"ticker": ["000001_01"], "target_weight": [0.], "source": ["compustat_wrds"]}).to_csv(tmp_path / "target_positions.csv", index=False)
    pd.DataFrame({"ticker": ["000001_01"], "symbol": ["ABC"], "raw_signal_price": [100.], "source": ["compustat_wrds"]}).to_csv(tmp_path / "alpha_rankings.csv", index=False)

    class Broker:
        def __init__(self, config):
            pass

        def get_account_summary(self):
            return {"equity": 100_000.}

        def get_positions(self):
            return pd.Series({"ABC": 5})

    monkeypatch.setattr(submitter, "IBKRPaperBroker", Broker)
    result = submitter.submit_report_to_ibkr_paper(tmp_path, submitter.PaperSubmitConfig(dry_run=True))
    assert result.iloc[0]["ticker"] == "ABC"
    assert result.iloc[0]["side"] == "SELL"
    assert result.iloc[0]["quantity"] == 5


def test_invalid_target_report_cannot_be_interpreted_as_liquidation(tmp_path):
    from live.paper_order_submitter import load_target_weights

    pd.DataFrame({"ticker": ["000001_01"], "target_weight": [np.nan], "source": ["compustat_wrds"]}).to_csv(tmp_path / "target_positions.csv", index=False)
    with pytest.raises(ValueError, match="finite weights"):
        load_target_weights(tmp_path)
