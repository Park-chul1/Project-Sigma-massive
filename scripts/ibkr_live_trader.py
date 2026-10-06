from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ib_insync import IB, MarketOrder, Stock
except ImportError as exc:
    raise ImportError(
        "ib_insync is required to run this script. Install it with `pip install ib_insync`."
    ) from exc


def parse_args():
    p = argparse.ArgumentParser(description="IBKR live trading helper for paper trading and pipeline target positions")
    p.add_argument("--host", default="172.30.1.41", help="TWS/Gateway host")
    p.add_argument("--port", type=int, default=7497, help="TWS demo socket port")
    p.add_argument("--client-id", type=int, default=1, help="IB API client ID")
    p.add_argument("--account", default=None, help="Optional IB account to filter account values")
    p.add_argument("--symbols", nargs="*", default=["AAPL"], help="Symbols for the optional sample order")
    p.add_argument("--load-pipeline", default=None, help="Directory containing saved pipeline outputs with tickers.csv and weights.npy or X.npy/factor_returns.npy for fallback calculation")
    p.add_argument("--date-index", type=int, default=-1, help="Index of the date to load target weights from weights.npy or compute from pipeline outputs")
    p.add_argument("--method", default="ewma", choices=["latest", "rolling", "ewma", "zero", "oracle"], help="Factor return prediction method when computing weights from saved pipeline outputs")
    p.add_argument("--lookback", type=int, default=20, help="Lookback window for factor return prediction")
    p.add_argument("--ewma-halflife", type=float, default=20.0, help="EWMA halflife for factor return prediction")
    p.add_argument("--min-periods", type=int, default=5, help="Minimum positive observations for factor return prediction")
    p.add_argument("--quantile", type=float, default=0.10, help="Quantile for long/short portfolio weights when computing from scores")
    p.add_argument("--gross", type=float, default=2.0, help="Gross exposure for long/short weight construction")
    p.add_argument("--notional", type=float, default=100000.0, help="Notional size for target weights when estimating order sizes")
    p.add_argument("--dry-run", action="store_true", help="Do not send live orders; only show what would be traded")
    p.add_argument("--place-sample-order", action="store_true", help="Place a sample market order for the first symbol")
    p.add_argument("--sample-qty", type=int, default=1, help="Quantity for the sample market order")
    return p.parse_args()


def connect_ibkr(host: str, port: int, client_id: int, timeout: float = 10.0) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=timeout)
    return ib


def stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    return Stock(symbol, exchange, currency)


from factor_pipeline.saved_targets import load_pipeline_target_weights


def compute_target_dollars(weights: np.ndarray, total_equity: float, gross: float = 2.0) -> np.ndarray:
    if gross <= 0:
        raise ValueError("gross must be positive")
    return weights * (total_equity / gross)


def print_account_summary(ib: IB, account: str | None = None) -> None:
    accounts = ib.managedAccounts()
    print("managedAccounts:", accounts)
    values = ib.accountValues()
    if account is not None:
        values = [v for v in values if v.account == account]
    print(f"accountValues ({len(values)})")
    for v in values[:50]:
        print(v)
    positions = ib.positions()
    print(f"positions ({len(positions)})")
    for pos in positions[:50]:
        print(pos)


def place_market_order(ib: IB, symbol: str, action: str, quantity: int) -> None:
    contract = stock_contract(symbol)
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    print(f"Placed {action} {quantity} {symbol}, order status={trade.orderStatus.status}")
    return


def main():
    args = parse_args()
    ib = connect_ibkr(args.host, args.port, args.client_id)
    print("connected:", ib.isConnected())
    print("managedAccounts:", ib.managedAccounts())

    print("--- account summary ---")
    print_account_summary(ib, account=args.account)

    if args.load_pipeline:
        pipeline_path = Path(args.load_pipeline)
        tickers, weights = load_pipeline_target_weights(
            pipeline_path,
            args.date_index,
            method=args.method,
            lookback=args.lookback,
            ewma_halflife=args.ewma_halflife,
            min_periods=args.min_periods,
            quantile=args.quantile,
            gross=args.gross,
        )
        print(f"Loaded {len(tickers)} tickers from {pipeline_path}")
        target_dollars = compute_target_dollars(weights, args.notional)
        nonzero = np.where(np.isfinite(target_dollars) & (target_dollars != 0.0))[0]
        print(f"Nonzero target positions: {len(nonzero)}")
        for idx in nonzero[:20]:
            print(f"{tickers[idx]} -> target ${target_dollars[idx]:.2f}")
        if len(nonzero) > 20:
            print("...")

    if args.place_sample_order:
        if args.dry_run:
            print("dry run: sample order not sent")
        else:
            symbol = args.symbols[0] if args.symbols else "AAPL"
            action = "BUY" if args.sample_qty > 0 else "SELL"
            place_market_order(ib, symbol, action, abs(args.sample_qty))

    ib.disconnect()
    print("disconnected")


if __name__ == "__main__":
    main()
