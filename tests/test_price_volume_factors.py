import pandas as pd

from factor_pipeline.backtest import portfolio_returns
from factor_pipeline.price_volume_factors import build_price_volume_factors
from factor_pipeline.panel import build_tradable_mask, compute_forward_returns


def test_price_volume_factor_names_exist():
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    close = pd.DataFrame({"A": range(100, 400)}, index=idx, dtype=float)
    panel = {"adj_close": close, "close": close, "open": close*0.99, "high": close*1.01, "low": close*0.98, "volume": close*1000}
    f = build_price_volume_factors(panel)
    assert "mom_252" in f
    assert "amihud_20" in f
    assert "dist_52w_high" in f
    assert not any(name.startswith("rev_") for name in f)


def test_forward_returns_are_future_returns():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    r = compute_forward_returns(close, horizon=1)
    assert abs(r.iloc[0, 0] - 0.10) < 1e-12
    assert abs(r.iloc[1, 0] - 0.10) < 1e-12
    assert pd.isna(r.iloc[2, 0])


def test_forward_returns_can_drop_outliers():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame({"A": [1.0, 50.0, 51.0]}, index=idx)
    r = compute_forward_returns(close, horizon=1, max_abs_return=1.0)
    assert pd.isna(r.iloc[0, 0])
    assert abs(r.iloc[1, 0] - 0.02) < 1e-12


def test_portfolio_returns_drop_outliers_before_coverage_check():
    weights = pd.DataFrame([[0.5, -0.5]]).to_numpy(dtype=float)
    returns = pd.DataFrame([[20.0, -0.02]]).to_numpy(dtype=float)
    port, coverage = portfolio_returns(weights, returns, min_return_coverage=0.5, max_abs_return=1.0)
    assert coverage[0] == 0.5
    assert abs(port[0] - 0.01) < 1e-12


def test_build_tradable_mask_requires_positive_price_and_volume():
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    panel = {
        "adj_close": pd.DataFrame({"A": [10.0, 11.0], "B": [5.0, None]}, index=idx),
        "volume": pd.DataFrame({"A": [100.0, 0.0], "B": [50.0, 60.0]}, index=idx),
    }

    mask = build_tradable_mask(panel)

    expected = pd.DataFrame({"A": [True, False], "B": [True, False]}, index=idx)
    pd.testing.assert_frame_equal(mask, expected)


def test_forward_delisting_recovery_uses_last_valid_price_within_horizon():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"A": [100.0, 120.0, 80.0, 999.0, 999.0]}, index=idx)
    metadata = pd.DataFrame({"ticker": ["A"], "delisted_utc": [idx[3]]})
    result = compute_forward_returns(close, horizon=3, ticker_metadata=metadata)
    assert abs(result.iloc[0, 0] - (-0.6)) < 1e-12
    assert abs(result.iloc[1, 0] - (40.0 / 120.0 - 1.0)) < 1e-12
    assert result.iloc[2:, 0].isna().all()  # No observed full horizon at the end.
    one_day = compute_forward_returns(close, ticker_metadata=metadata, max_abs_return=0.1)
    assert abs(one_day.iloc[2, 0] - (-0.5)) < 1e-12
    assert one_day.iloc[3:, 0].isna().all()  # Never repeat settlement.


def test_forward_returns_do_not_guess_delisting_from_missing_quotes():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 120.0, None, None]}, index=idx)
    result = compute_forward_returns(close)
    assert result.iloc[1:, 0].isna().all()
    metadata = pd.DataFrame({"ticker": ["A"], "delisted_utc": [idx[2]]})
    recovered = compute_forward_returns(close, ticker_metadata=metadata)
    assert recovered.iloc[1, 0] == -0.5


def test_confirmed_terminal_returns_survive_research_outlier_filter():
    weights = pd.DataFrame([[0.5, -0.5]]).to_numpy(dtype=float)
    returns = pd.DataFrame([[-0.5, -0.5]]).to_numpy(dtype=float)
    terminal = pd.DataFrame([[True, True]]).to_numpy(dtype=bool)
    port, coverage = portfolio_returns(weights, returns, max_abs_return=0.1, terminal_return_mask=terminal)
    assert coverage[0] == 1.0
    assert port[0] == 0.0
    ordinary_port, ordinary_coverage = portfolio_returns(weights, returns, max_abs_return=0.1)
    assert ordinary_coverage[0] == 0.0
    assert pd.isna(ordinary_port[0])


def test_terminal_return_mask_rejects_wrong_shape_and_cannot_fill_missing_price():
    import numpy as np
    import pytest

    weights = np.array([[1.0]])
    returns = np.array([[np.nan]])
    with pytest.raises(ValueError, match="terminal_return_mask"):
        portfolio_returns(weights, returns, terminal_return_mask=np.zeros((2, 1), dtype=bool))
    port, coverage = portfolio_returns(weights, returns, terminal_return_mask=np.ones((1, 1), dtype=bool))
    assert coverage[0] == 0.0
    assert np.isnan(port[0])
