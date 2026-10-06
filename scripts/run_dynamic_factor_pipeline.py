from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.estimation import estimate_factor_returns
from factor_pipeline.universe import load_wrds_universe_mask
from factors.kalman_filter import kalman_filter_factor_returns_from_ols
from factors.regime_sensor import classify_regime
from factors.blend import blend_factor_returns
from factors.dynamic_diagnostics import dynamic_factor_diagnostics, print_dynamic_summary


def load_pipeline_data(out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load X, r, f_ols, dates, tickers, factor_names from clean pipeline output."""
    X = np.load(out_dir / "X.npy")
    r = np.load(out_dir / "r.npy")
    f_ols = np.load(out_dir / "factor_returns.npy")
    dates_df = pd.read_csv(out_dir / "dates.csv")
    dates = pd.to_datetime(dates_df["date"]).to_numpy()
    tickers_df = pd.read_csv(out_dir / "tickers.csv")
    tickers = tickers_df["ticker"].tolist()
    policy_mask = load_wrds_universe_mask(out_dir, pd.DatetimeIndex(dates), tickers).to_numpy(dtype=bool)
    saved_mask = np.load(out_dir / "tradable_mask.npy")
    if saved_mask.shape != policy_mask.shape or (saved_mask & ~policy_mask).any():
        raise ValueError("Input tradable mask violates the WRDS NASDAQ market-cap policy; rebuild inputs")
    factor_names_df = pd.read_csv(out_dir / "factor_names.csv")
    factor_names = factor_names_df["factor"].tolist()
    return X, r, f_ols, dates, tickers, factor_names


def save_factor_returns(out_dir: Path, dates: np.ndarray, factor_names: list[str], f: np.ndarray, name: str) -> None:
    """Save factor returns to CSV."""
    df = pd.DataFrame(f, index=dates, columns=factor_names)
    df.index.name = "date"
    df.to_csv(out_dir / f"factor_returns_{name}.csv")


def save_array(out_dir: Path, dates: np.ndarray, data: np.ndarray, name: str) -> None:
    """Save 1D array to CSV."""
    df = pd.DataFrame({"date": dates, name: data})
    df.to_csv(out_dir / f"{name}.csv", index=False)


def save_regime_thresholds(out_dir: Path, dates: np.ndarray, warning_threshold: np.ndarray, break_threshold: np.ndarray) -> None:
    """Save regime thresholds."""
    df = pd.DataFrame({
        "date": dates,
        "warning_threshold": warning_threshold,
        "break_threshold": break_threshold,
    })
    df.to_csv(out_dir / "regime_thresholds.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Dynamic factor estimation pipeline")
    parser.add_argument("--input-dir", default="data/processed_wrds", help="Input directory with WRDS pipeline data")
    parser.add_argument("--output-dir", default="data/processed_dynamic_wrds", help="Output directory for dynamic results")
    parser.add_argument("--q-scale", type=float, default=1e-4, help="Kalman process noise scale")
    parser.add_argument("--r-scale", type=float, default=1e-2, help="Kalman measurement noise scale")
    parser.add_argument("--p0-scale", type=float, default=1.0, help="Kalman initial covariance scale")
    parser.add_argument("--lookback", type=int, default=252, help="Regime sensor lookback window")
    parser.add_argument("--warning-q", type=float, default=0.95, help="Warning quantile threshold")
    parser.add_argument("--break-q", type=float, default=0.99, help="Break quantile threshold")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clean pipeline data...")
    X, r, f_ols, dates, tickers, factor_names = load_pipeline_data(input_dir)

    print("Running Kalman filter...")
    print("---------------- Kalman Filter ----------------")
    print("mode: factor-level OLS observation")
    print(f"input f_ols shape: {f_ols.shape}")
    print("max Kalman matrix size: K x K")
    print(f"q_scale: {args.q_scale}")
    print(f"r_scale: {args.r_scale}")
    kf_result = kalman_filter_factor_returns_from_ols(
        f_ols,
        q_scale=args.q_scale,
        r_scale=args.r_scale,
        p0_scale=args.p0_scale,
    )
    f_kf = kf_result["f_kf"]
    innovation_score = kf_result["innovation_score"]
    innovation_norm = kf_result["innovation_norm"]
    valid_factor_count = kf_result["valid_factor_count"]
    print(f"valid factor obs mean: {valid_factor_count.mean():.1f}")
    print(f"skipped dates: {(valid_factor_count == 0).sum()}")
    print(f"innovation NaN count: {np.isnan(innovation_score).sum()}")
    print("----------------------------------------------")

    print("Classifying regimes...")
    regime_result = classify_regime(
        innovation_score,
        lookback=args.lookback,
        warning_q=args.warning_q,
        break_q=args.break_q,
    )
    regime_states = regime_result["regime_states"]
    warning_threshold = regime_result["warning_threshold"]
    break_threshold = regime_result["break_threshold"]

    print("Blending factor returns...")
    f_blend, alpha_ols = blend_factor_returns(f_ols, f_kf, regime_states)

    print("Saving results...")
    save_factor_returns(output_dir, dates, factor_names, f_ols, "ols")
    save_factor_returns(output_dir, dates, factor_names, f_kf, "kalman")
    save_factor_returns(output_dir, dates, factor_names, f_blend, "blend")

    save_array(output_dir, dates, innovation_score, "innovation_scores")
    save_array(output_dir, dates, innovation_norm, "innovation_norm")
    save_array(output_dir, dates, regime_states, "regime_states")
    save_array(output_dir, dates, alpha_ols, "blend_alpha_ols")
    save_array(output_dir, dates, valid_factor_count, "valid_factor_count")
    save_regime_thresholds(output_dir, dates, warning_threshold, break_threshold)

    # Save arrays needed by backtests and live portfolio rebalancing.
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "factor_returns.npy", f_blend)  # Use blended returns for portfolio
    np.save(output_dir / "r.npy", r)
    for name in ("weights.npy", "positions.npy", "scores.npy", "f_pred.npy"):
        (output_dir / name).unlink(missing_ok=True)

    # Copy static files from the selected input directory.
    import shutil
    for name in ("nasdaq_membership.parquet", "daily_bars.parquet", "pipeline_summary.json", "terminal_return_mask.npy"):
        source = input_dir / name
        destination = output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy(source, destination)
    shutil.copy(input_dir / "tickers.csv", output_dir / "tickers.csv")
    shutil.copy(input_dir / "factor_names.csv", output_dir / "factor_names.csv")
    dates_path = input_dir / "dates.csv"
    if dates_path.exists():
        shutil.copy(dates_path, output_dir / "dates.csv")
    tradable_path = input_dir / "tradable_mask.npy"
    if tradable_path.exists():
        shutil.copy(tradable_path, output_dir / "tradable_mask.npy")

    print("Generating diagnostics...")
    diag = dynamic_factor_diagnostics(
        dates=dates,
        tickers=tickers,
        factor_names=factor_names,
        f_ols=f_ols,
        f_kf=f_kf,
        f_blend=f_blend,
        innovation_score=innovation_score,
        innovation_norm=innovation_norm,
        regime_states=regime_states,
        warning_threshold=warning_threshold,
        break_threshold=break_threshold,
        valid_obs_count=valid_factor_count,
        alpha_ols=alpha_ols,
        q_scale=args.q_scale,
        r_scale=args.r_scale,
        lookback=args.lookback,
        warning_q=args.warning_q,
        break_q=args.break_q,
    )

    # Save summary
    import json
    with open(output_dir / "dynamic_factor_summary.json", "w") as f:
        json.dump(diag, f, indent=2, default=str)

    print_dynamic_summary(diag)

    # Anti-lookahead comment
    print("\n" + "="*50)
    print("ANTI-LOOKAHEAD RULE:")
    print("Regime states are computed using innovation_score[t], which depends on returns at date t.")
    print("For trading, use regime_for_trading[t] = regime_states[t - 1] to avoid lookahead bias.")
    print("="*50)


if __name__ == "__main__":
    main()
