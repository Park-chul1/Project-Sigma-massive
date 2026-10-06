"""Exercise the actual pipeline entry point with deterministic WRDS fixtures."""
from contextlib import nullcontext
import json
import sys

import numpy as np
import pandas as pd

from scripts import run_clean_pipeline
from factor_pipeline.fundamental_factors import build_fundamental_factors


def test_pipeline_writes_aligned_nasdaq_policy_and_delisting_returns(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=65)
    ids = [f"{i + 1:06d}_01" for i in range(10)]
    membership = pd.DataFrame({
        "ticker": ids, "from_date": pd.Timestamp("2020-01-01"),
        "thru_date": pd.NaT, "source": "compustat_wrds",
    })
    metadata = pd.DataFrame({
        "ticker": ids, "symbol": [f"S{i}" for i in range(10)],
        "delisted_utc": [pd.NaT] * 9 + [dates[50]], "source": "compustat_wrds",
    }).iloc[::-1].reset_index(drop=True)
    rows = []
    for day, date in enumerate(dates):
        for i, ticker in enumerate(ids):
            price = 30 + i + day * (i + 1) * .01 + np.sin(day * .1 + i) * .2
            rows.append({
                "date": date, "ticker": ticker, "open": price * .995,
                "high": price * 1.01, "low": price * .98, "close": price,
                "adj_close": price, "raw_close": price * 2, "volume": 100_000 + 100 * day + i,
                "market_cap": (i + 1) * 1e8, "source": "compustat_wrds",
            })
    bars = pd.DataFrame(rows)
    # The same names change market-cap rank without changing their identities.
    bars.loc[bars["date"].ge(dates[30]) & bars["ticker"].eq(ids[0]), "market_cap"] = 2e9
    monkeypatch.setattr(run_clean_pipeline, "WRDSClient", lambda **kwargs: nullcontext(object()))
    monkeypatch.setattr(run_clean_pipeline, "download_nasdaq_membership", lambda *args: membership)
    monkeypatch.setattr(run_clean_pipeline, "download_security_metadata", lambda *args: metadata)
    monkeypatch.setattr(run_clean_pipeline, "download_daily_bars", lambda *args: bars)
    monkeypatch.setattr(run_clean_pipeline, "download_fundamentals", lambda *args, **kwargs: pd.DataFrame())
    out = tmp_path / "processed"
    out.mkdir()
    np.save(out / "weights.npy", np.ones((len(dates), len(ids))))
    monkeypatch.setattr(sys, "argv", [
        "run_clean_pipeline.py", "--start", str(dates[0].date()), "--end", str(dates[-1].date()),
        "--out-dir", str(out), "--cache-dir", str(tmp_path / "cache"),
        "--min-names", "3", "--max-factor-corr", "0",
    ])
    run_clean_pipeline.main()
    assert pd.read_csv(out / "tickers.csv")["ticker"].tolist() == ids
    mask = np.load(out / "tradable_mask.npy")
    assert not mask[:30, :2].any()
    assert mask[30:, 0].all()
    assert not mask[30:, 1:3].any()
    assert not mask[50:, 9].any()
    returns = np.load(out / "r.npy")
    assert returns[49, 9] == -.5
    assert np.isnan(returns[50:, 9]).all()
    summary = json.loads((out / "pipeline_summary.json").read_text())
    assert summary["data_policy"]["source"] == "compustat_wrds"
    assert summary["data_policy"]["bottom_market_cap_fraction"] == .2
    assert (out / "nasdaq_membership.parquet").exists()
    assert (out / "daily_bars.parquet").exists()
    assert not (out / "weights.npy").exists()
    assert np.load(out / "terminal_return_mask.npy")[49, 9]


def test_fundamental_valuation_uses_daily_cap_instead_of_split_adjusted_price():
    index = pd.DatetimeIndex(["2024-01-02"])
    fund = {"net_income": pd.DataFrame({"A": [20.]}, index=index),
            "shares_diluted": pd.DataFrame({"A": [10.]}, index=index)}
    close = pd.DataFrame({"A": [5.]}, index=index)
    cap = pd.DataFrame({"A": [100.]}, index=index)
    factors = build_fundamental_factors(fund, close, market_cap=cap)
    assert factors["earnings_yield"].iloc[0, 0] == .2
