from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.wrds_client import (
    WRDSClient, download_nasdaq_membership, download_security_metadata,
    download_daily_bars, download_fundamentals,
)
from factor_pipeline.universe import (
    build_nasdaq_membership_mask, exclude_bottom_market_cap, validate_wrds_source,
)
from factor_pipeline.panel import bars_long_to_panel, build_tradable_mask, compute_forward_returns, delisting_dates
from factor_pipeline.price_volume_factors import build_price_volume_factors
from factor_pipeline.fundamental_factors import build_ttm_financials, fundamentals_to_daily, build_fundamental_factors
from factor_pipeline.preprocess import apply_universe_mask, build_exposure_tensor
from factor_pipeline.estimation import estimate_factor_returns
from factor_pipeline.diagnostics import array_summary, factor_diagnostics, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Compustat/WRDS historical NASDAQ price+fundamental factor pipeline")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--wrds-username", default=None, help="WRDS username; defaults to WRDS_USERNAME")
    p.add_argument("--out-dir", default="data/processed_wrds")
    p.add_argument("--cache-dir", default="data/cache_wrds")
    p.add_argument("--min-names", type=int, default=30)
    p.add_argument("--ridge", default="1e-4",
                   help="Fixed ridge lambda, or 'auto' to select lambda date-by-date with GCV")
    p.add_argument("--ridge-grid", default=None,
                   help="Comma-separated lambda grid for --ridge auto, e.g. 1e-6,1e-5,1e-4,1e-3")
    p.add_argument("--ridge-solver", default="qr", choices=["qr", "normal"],
                   help="qr solves the augmented ridge least-squares system")
    p.add_argument("--estimation-workers", type=int, default=1,
                   help="Parallel worker threads for date-by-date factor return estimation")
    p.add_argument("--horizon", type=int, default=1, choices=[1],
                   help="Daily factor forecasts require a one-session return horizon")
    p.add_argument(
        "--max-abs-forward-return",
        type=float,
        default=1.0,
        help="Drop forward returns whose absolute value exceeds this threshold; use <=0 to disable",
    )
    p.add_argument("--financial-lookback-days", type=int, default=550,
                   help="Extra report-period history before --start used to seed quarterly TTM values")
    p.add_argument("--financial-lag-days", type=int, default=60, help="Availability lag when Compustat rdq is missing")
    p.add_argument("--ttm-min-quarters", type=int, default=4,
                   help="Minimum quarterly rows required to build a TTM flow value")
    p.add_argument("--min-factor-coverage", type=float, default=0.02, help="Drop factors with lower processed finite coverage")
    p.add_argument("--max-factor-corr", type=float, default=0.999,
                   help="Drop later factors whose abs correlation with an earlier kept factor is at least this value; use 0 to disable")
    p.add_argument("--corr-min-overlap", type=int, default=100,
                   help="Minimum finite pair observations required before applying the factor correlation filter")
    p.add_argument("--no-fill-missing-exposures", action="store_true", help="Keep NaNs in X after preprocessing instead of neutral 0 fill")
    p.add_argument("--no-cache", action="store_true", help="Disable parquet dataset caches")
    p.add_argument("--run-diagnostics", action="store_true", help="Run post-pipeline ridge diagnostics")
    p.add_argument("--diagnostics-lambdas", default="0,1e-8,1e-6,1e-4,1e-2,1e-1",
                   help="Comma-separated lambda values for diagnostics")
    p.add_argument("--diagnostics-output-dir", default=None,
                   help="Output directory for diagnostics. Default: <out-dir>/diagnostics")
    return p.parse_args()

def parse_ridge(value: str) -> float | str:
    return "auto" if str(value).lower() == "auto" else float(value)


def parse_ridge_grid(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    args = parse_args()
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise ValueError("--start must be on or before --end")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.financial_lag_days < 1 or args.financial_lookback_days < 0:
        raise ValueError("financial lag must be positive and lookback nonnegative")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    financial_report_start = (
        pd.Timestamp(args.start) - pd.Timedelta(days=args.financial_lookback_days)
    ).date().isoformat()
    with WRDSClient(
        wrds_username=args.wrds_username,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    ) as client:
        membership = download_nasdaq_membership(client, args.start, args.end)
        validate_wrds_source(membership, "NASDAQ membership")
        tickers_df = download_security_metadata(client, membership)
        validate_wrds_source(tickers_df, "Security metadata")
        tickers_df = tickers_df.sort_values("ticker").reset_index(drop=True)
        tickers = tickers_df["ticker"].astype(str).tolist()
        if len(tickers) != len(set(tickers)):
            raise ValueError("WRDS security identifiers must be unique")
        bars = download_daily_bars(client, membership, args.start, args.end)
        validate_wrds_source(bars, "Daily bars")
        fin_flat = download_fundamentals(
            client, membership, financial_report_start, args.end,
            lag_days=args.financial_lag_days,
        )

    panel = bars_long_to_panel(bars, tickers=tickers)
    dates = panel["adj_close"].index
    caps = bars.pivot(index="date", columns="ticker", values="market_cap")
    caps.index = pd.to_datetime(caps.index).normalize()
    members = build_nasdaq_membership_mask(membership, dates, tickers)
    cap_mask = exclude_bottom_market_cap(caps, members)
    tradable_mask_df = build_tradable_mask(panel) & cap_mask
    for row in tickers_df.itertuples(index=False):
        delisted = pd.to_datetime(getattr(row, "delisted_utc", None), errors="coerce")
        if pd.notna(delisted):
            tradable_mask_df.loc[dates >= delisted, row.ticker] = False
    tradable_mask = tradable_mask_df.to_numpy(dtype=bool)
    if not tradable_mask.any():
        raise ValueError("No eligible NASDAQ securities after market-cap and tradability filters")

    membership.to_parquet(out_dir / "nasdaq_membership.parquet", index=False)
    bars.to_parquet(out_dir / "daily_bars.parquet", index=False)
    if not fin_flat.empty:
        fin_flat.to_parquet(out_dir / "financials_flat.parquet", index=False)
    else:
        (out_dir / "financials_flat.parquet").unlink(missing_ok=True)
    fin_model = build_ttm_financials(fin_flat, min_quarters=args.ttm_min_quarters)
    if not fin_model.empty:
        fin_model.to_parquet(out_dir / "financials_ttm.parquet", index=False)
    else:
        (out_dir / "financials_ttm.parquet").unlink(missing_ok=True)
    fund_daily = fundamentals_to_daily(fin_model, dates, tickers) if not fin_model.empty else {}

    pv_factors = build_price_volume_factors(panel)
    fundamental_factors = build_fundamental_factors(
        fund_daily, panel["raw_close"], market_cap=caps.reindex(index=dates, columns=tickers),
    ) if fund_daily else {}
    factors = apply_universe_mask({**pv_factors, **fundamental_factors}, tradable_mask_df)

    X, factor_names, preprocess_diag = build_exposure_tensor(
        factors,
        min_names=args.min_names,
        min_factor_coverage=args.min_factor_coverage,
        fill_missing=not args.no_fill_missing_exposures,
        max_factor_corr=args.max_factor_corr,
        corr_min_overlap=args.corr_min_overlap,
    )
    max_abs_forward_return = args.max_abs_forward_return if args.max_abs_forward_return > 0 else None
    r_df = compute_forward_returns(
        panel["adj_close"],
        horizon=args.horizon,
        max_abs_return=max_abs_forward_return,
        ticker_metadata=tickers_df,
    ).where(tradable_mask_df)
    r = r_df.to_numpy(dtype=float)
    terminal_return_mask = np.zeros_like(tradable_mask)
    for i, event in enumerate(delisting_dates(tickers_df, tickers)):
        if pd.notna(event):
            terminal_return_mask[:-args.horizon, i] = (
                (dates[:-args.horizon] < event) & (dates[args.horizon:] >= event)
            )
    terminal_return_mask &= np.isfinite(r)
    ridge_value = parse_ridge(args.ridge)
    f, ridge_diag = estimate_factor_returns(
        X,
        r,
        min_names=args.min_names,
        ridge=ridge_value,
        universe_mask=tradable_mask,
        ridge_grid=parse_ridge_grid(args.ridge_grid),
        ridge_selection="gcv" if ridge_value == "auto" else "fixed",
        solver=args.ridge_solver,
        n_jobs=args.estimation_workers,
        return_diagnostics=True,
    )

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "r.npy", r)
    np.save(out_dir / "factor_returns.npy", f)
    np.save(out_dir / "tradable_mask.npy", tradable_mask)
    np.save(out_dir / "terminal_return_mask.npy", terminal_return_mask)
    # Old targets may have the same dimensions after a source/ranking refresh.
    # They must be regenerated from the newly estimated factors.
    for name in ("weights.npy", "positions.npy", "scores.npy", "f_pred.npy"):
        (out_dir / name).unlink(missing_ok=True)
    tickers_df.to_csv(out_dir / "tickers.csv", index=False)
    pd.DataFrame({"date": dates}).to_csv(out_dir / "dates.csv", index=False)
    pd.DataFrame({"factor": factor_names}).to_csv(out_dir / "factor_names.csv", index=False)
    factor_diagnostics(factors).to_csv(out_dir / "factor_diagnostics_raw.csv", index=False)
    preprocess_diag.to_csv(out_dir / "factor_diagnostics_preprocessed.csv", index=False)
    ridge_diag.insert(0, "date", dates.to_numpy())
    ridge_diag.to_csv(out_dir / "ridge_diagnostics.csv", index=False)

    selected = ridge_diag["selected_ridge"].replace([np.inf, -np.inf], np.nan).dropna()
    ridge_summary = {
        "mode": "gcv" if ridge_value == "auto" else "fixed",
        "solver": args.ridge_solver,
        "fixed_lambda": None if ridge_value == "auto" else float(ridge_value),
        "grid": parse_ridge_grid(args.ridge_grid),
        "finite_dates": int(selected.size),
        "median_selected_lambda": float(selected.median()) if not selected.empty else None,
        "min_selected_lambda": float(selected.min()) if not selected.empty else None,
        "max_selected_lambda": float(selected.max()) if not selected.empty else None,
    }

    summary = {
        "data_policy": {
            "source": "compustat_wrds",
            "prices": "Compustat sec_dprc through WRDS; split-adjusted daily OHLC/volume with separate raw prices and same-day market_cap.",
            "fundamentals": "Compustat fundq; reported availability plus lag fallback; quarterly flows converted to trailing four quarters. Standard Compustat may contain restatements and is not a vintage filing database.",
            "look_ahead_bias_control": "Historical NASDAQ exchange intervals, same-date capitalization, and backward-only financial availability; no future ranking window.",
            "universe": "All Compustat-covered historical NASDAQ securities (EXCHG=14); no current-active-only filter or fixed top-N selection.",
            "bottom_market_cap_fraction": 0.20,
            "tradable_mask": "Per-date membership, positive market cap excluding bottom ceil(20%), positive price/volume, and no confirmed inactivation on/before the signal date.",
            "delisting_recovery_fraction": 0.50,
            "delisting": "Compustat security inactivation date/reason is the termination proxy; recover 50% of the final valid pre-event price. Index/exchange departures and missing quotes alone are not termination events.",
            "cache": "Only query-keyed Compustat/WRDS caches; old vendor caches are never reused.",
        },
        "params": vars(args),
        "n_dates": len(dates),
        "n_tickers": len(tickers),
        "n_membership_intervals": int(len(membership)),
        "n_financial_rows_flat": int(len(fin_flat)),
        "n_financial_rows_model": int(len(fin_model)),
        "n_factors_raw": len(factors),
        "n_factors_kept": len(factor_names),
        "n_factors": len(factor_names),
        "max_abs_forward_return": max_abs_forward_return,
        "price_volume_factors": list(pv_factors.keys()),
        "fundamental_factors": list(fundamental_factors.keys()),
        "X": array_summary("X", X),
        "r": array_summary("r", r),
        "factor_returns": array_summary("factor_returns", f),
        "ridge": ridge_summary,
        "estimation_workers": args.estimation_workers,
        "tradable_mask": {
            "shape": list(tradable_mask.shape),
            "true_count": int(tradable_mask.sum()),
            "true_ratio": float(tradable_mask.mean()) if tradable_mask.size else 0.0,
        },
        "factor_filtering": {
            "min_factor_coverage": args.min_factor_coverage,
            "max_factor_corr": args.max_factor_corr,
            "corr_min_overlap": args.corr_min_overlap,
            "fill_missing_exposures_after_zscore": not args.no_fill_missing_exposures,
            "kept_factors": factor_names,
            "dropped_factors": preprocess_diag.loc[~preprocess_diag["kept"], "factor"].tolist() if not preprocess_diag.empty else [],
            "high_corr_dropped_factors": preprocess_diag.loc[preprocess_diag["drop_reason"].eq("high_corr"), "factor"].tolist() if not preprocess_diag.empty else [],
        },
    }
    save_json(summary, out_dir / "pipeline_summary.json")
    print(f"done: {out_dir}")
    print(f"X={X.shape}, r={r.shape}, factor_returns={f.shape}, factors={len(factor_names)}")

    # Optional: run post-pipeline ridge diagnostics
    if args.run_diagnostics:
        from factor_pipeline.diagnostics_ridge import run_estimation_diagnostics
        diagnostics_out = args.diagnostics_output_dir or str(out_dir / "diagnostics")
        lambdas = [float(x.strip()) for x in args.diagnostics_lambdas.split(",")]
        run_estimation_diagnostics(
            X=X,
            r=r,
            valid_mask=tradable_mask,
            lambdas=lambdas,
            factor_names=factor_names,
            output_dir=diagnostics_out,
            min_names=args.min_names,
        )


if __name__ == "__main__":
    main()
