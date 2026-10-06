from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_pipeline.backtest import run_factor_backtest
from factor_pipeline.diagnostics import save_json, array_summary
from factor_pipeline.universe import load_wrds_universe_mask


def parse_args():
    p = argparse.ArgumentParser(description="Run long-short backtest from saved factor pipeline outputs")
    p.add_argument("--input-dir", default="data/processed_wrds")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--method", default="latest", choices=["latest", "rolling", "ewma", "zero", "oracle"])
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--ewma-halflife", type=float, default=20.0)
    p.add_argument("--min-periods", type=int, default=5)
    p.add_argument("--quantile", type=float, default=0.10, help="0.10 means top 10%% long and bottom 10%% short")
    p.add_argument("--gross", type=float, default=2.0, help="2.0 means long +1 and short -1")
    p.add_argument("--min-names-per-side", type=int, default=5)
    p.add_argument("--min-return-coverage", type=float, default=0.80)
    p.add_argument(
        "--max-abs-return",
        type=float,
        default=1.0,
        help="Drop realized stock returns whose absolute value exceeds this threshold; use <=0 to disable",
    )
    p.add_argument("--save-arrays", action="store_true", help="Also save scores.npy and positions.npy")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir / f"backtest_{args.method}_q{args.quantile:g}_lb{args.lookback}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(input_dir / "X.npy")
    r = np.load(input_dir / "r.npy")
    f = np.load(input_dir / "factor_returns.npy")
    tradable_mask = np.load(input_dir / "tradable_mask.npy")

    dates_path = input_dir / "dates.csv"
    dates = pd.read_csv(dates_path)["date"]
    tickers = pd.read_csv(input_dir / "tickers.csv", dtype={"ticker": str})["ticker"].tolist()
    policy_mask = load_wrds_universe_mask(input_dir, pd.DatetimeIndex(pd.to_datetime(dates)), tickers)
    tradable_mask &= policy_mask.to_numpy(dtype=bool)

    result = run_factor_backtest(
        X=X,
        r=r,
        f=f,
        tradable_mask=tradable_mask,
        terminal_return_mask=np.load(input_dir / "terminal_return_mask.npy"),
        method=args.method,
        lookback=args.lookback,
        ewma_halflife=args.ewma_halflife,
        min_periods=args.min_periods,
        quantile=args.quantile,
        gross=args.gross,
        min_names_per_side=args.min_names_per_side,
        min_return_coverage=args.min_return_coverage,
        max_abs_return=args.max_abs_return if args.max_abs_return > 0 else None,
    )

    daily = result["daily"].copy()
    daily.insert(0, "date", dates.iloc[: len(daily)].to_numpy())
    daily.to_csv(out_dir / "backtest_returns.csv", index=False)

    if args.save_arrays:
        np.save(out_dir / "scores.npy", result["scores"])
        np.save(out_dir / "positions.npy", result["weights"])
        np.save(out_dir / "f_pred.npy", result["f_pred"])

    summary = {
        "warning": "method=oracle intentionally uses same-date realized factor returns and is look-ahead only. Do not treat oracle as tradable.",
        "input_dir": str(input_dir),
        "params": vars(args),
        "metrics": result["metrics"],
        "X": array_summary("X", X),
        "r": array_summary("r", r),
        "factor_returns": array_summary("factor_returns", f),
        "f_pred": array_summary("f_pred", result["f_pred"]),
        "scores": array_summary("scores", result["scores"]),
        "positions": array_summary("positions", result["weights"]),
    }
    save_json(summary, out_dir / "backtest_summary.json")

    print(f"done: {out_dir}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
