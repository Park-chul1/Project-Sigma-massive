from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live.broker import IBKRPaperBroker
from live.config import LiveConfig
from factor_pipeline.universe import validate_wrds_source
from live.execution import generate_orders
from live.risk import apply_risk_checks


@dataclass(frozen=True)
class PaperSubmitConfig:
    host: str = "172.30.1.41"
    port: int = 7497
    client_id: int = 72
    account: str | None = None
    dry_run: bool = False
    max_orders: int | None = None
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0
    min_net_alpha_after_cost_bps: float = 5.0
    min_price: float = 0.0
    submit_mode: str = "burst"
    submit_pause_seconds: float = 0.05
    post_submit_wait_seconds: float = 5.0


def load_target_weights(report_dir: Path, cfg: PaperSubmitConfig | None = None) -> pd.Series:
    path = report_dir / "target_positions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing target positions file: {path}")
    df = pd.read_csv(path, dtype={"ticker": str})
    validate_wrds_source(df, "target portfolio report")
    required = {"ticker", "target_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.assign(target_weight=pd.to_numeric(df["target_weight"], errors="coerce"))
    if df["ticker"].isna().any() or df["ticker"].duplicated().any() or not np.isfinite(df["target_weight"]).all():
        raise ValueError("Target portfolio report requires unique security IDs and finite weights")
    if cfg is not None and "alpha_score" in df.columns:
        alpha = pd.to_numeric(df["alpha_score"], errors="coerce")
        threshold = (cfg.transaction_cost_bps + cfg.slippage_bps + cfg.min_net_alpha_after_cost_bps) / 10_000.0
        directional_alpha = alpha.where(df["target_weight"].ge(0), -alpha)
        keep = df["target_weight"].abs().le(1e-12) | directional_alpha.ge(threshold)
        df = df[keep].copy()
    if cfg is not None and cfg.min_price > 0 and "price_flag" in df.columns:
        price_flag = df["price_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
        keep = df["target_weight"].abs().le(1e-12) | price_flag
        df = df[keep].copy()
    weights = (
        df
        .dropna(subset=["ticker", "target_weight"])
        .groupby("ticker")["target_weight"]
        .last()
        .astype(float)
    )
    return weights[weights.abs() > 1e-12].sort_index()


def _apply_min_price_to_targets(target_weights: pd.Series, prices: pd.Series, min_price: float) -> pd.Series:
    if min_price <= 0 or target_weights.empty:
        return target_weights
    target_prices = pd.to_numeric(prices.reindex(target_weights.index), errors="coerce")
    too_low = target_prices.notna() & target_prices.lt(min_price)
    return target_weights.loc[~too_low].sort_index()


def _report_execution_inputs(report_dir: Path, model_ids: list[str]) -> tuple[pd.Series, pd.Series]:
    path = report_dir / "alpha_rankings.csv"
    frame = pd.read_csv(path, dtype={"ticker": str, "symbol": str})
    validate_wrds_source(frame, "daily order report")
    required = {"ticker", "symbol", "raw_signal_price"}
    if not required.issubset(frame.columns):
        raise ValueError("Regenerate the WRDS daily report with broker symbols and raw_signal_price")
    selected = frame[frame["ticker"].isin(model_ids)].copy()
    if selected["ticker"].duplicated().any() or selected["symbol"].duplicated().any():
        raise ValueError("Ambiguous Compustat-to-broker symbol mapping in daily report")
    mapping = selected.set_index("ticker")["symbol"].reindex(model_ids)
    if mapping.isna().any() or mapping.str.strip().eq("").any():
        raise ValueError("Missing broker symbols in WRDS daily report; refusing to submit security IDs")
    prices = pd.to_numeric(frame["raw_signal_price"], errors="coerce")
    valid = frame["symbol"].notna() & np.isfinite(prices) & prices.gt(0)
    price_frame = pd.DataFrame({"symbol": frame.loc[valid, "symbol"], "price": prices[valid]})
    duplicated = price_frame["symbol"].duplicated(keep=False)
    price_frame = price_frame.loc[~duplicated]
    return mapping, price_frame.set_index("symbol")["price"]


def _live_config(cfg: PaperSubmitConfig) -> LiveConfig:
    base = LiveConfig(
        DATA_PROVIDER="wrds",
        BROKER="ibkr_paper",
        PAPER_TRADING=True,
        ENABLE_REAL_TRADING=False,
        IBKR_HOST=cfg.host,
        IBKR_PORT=cfg.port,
        IBKR_BROKER_CLIENT_ID=cfg.client_id,
        IBKR_ACCOUNT=cfg.account,
        DRY_RUN=cfg.dry_run,
        COST_BPS=cfg.transaction_cost_bps,
        SLIPPAGE_BPS=cfg.slippage_bps,
        MIN_NET_ALPHA_AFTER_COST_BPS=cfg.min_net_alpha_after_cost_bps,
    )
    base.validate()
    return base


def submit_report_to_ibkr_paper(report_dir: str | Path, cfg: PaperSubmitConfig) -> pd.DataFrame:
    report_path = Path(report_dir)
    target_weights = load_target_weights(report_path, cfg)
    mapping, report_prices = _report_execution_inputs(report_path, target_weights.index.tolist())
    target_weights = target_weights.rename(index=mapping.to_dict())
    live_config = _live_config(cfg)
    broker = IBKRPaperBroker(live_config)
    rows: list[dict[str, Any]] = []
    try:
        account = broker.get_account_summary()
        equity = float(account.get("equity", live_config.PAPER_STARTING_EQUITY))
        current_positions = broker.get_positions()
        symbols = sorted(set(target_weights.index.astype(str)) | set(current_positions.index.astype(str)))
        prices = report_prices.reindex(symbols)
        if prices.isna().any():
            raise ValueError(f"Missing unadjusted WRDS prices for broker symbols: {prices[prices.isna()].index.tolist()}")
        target_weights = _apply_min_price_to_targets(target_weights, prices, cfg.min_price)

        orders = generate_orders(current_positions, target_weights, prices, equity, live_config)
        safe_orders, risk_rows = apply_risk_checks(orders, live_config, equity, prices)
        risk_by_key = {(r["ticker"], r["side"], int(r["quantity"])): r for r in risk_rows}
        if cfg.max_orders is not None:
            safe_orders = safe_orders[: cfg.max_orders]

        submitted_at = pd.Timestamp.utcnow().isoformat()

        def base_row(order):
            row = {
                "submitted_at": submitted_at,
                "ticker": order.ticker,
                "side": order.side,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
                "notional": abs(order.quantity * order.limit_price),
                "dry_run": cfg.dry_run,
                "order_status": "dry_run" if cfg.dry_run else "submitted",
                "order_id": "",
                "error": "",
                "transaction_cost_bps": cfg.transaction_cost_bps,
                "slippage_bps": cfg.slippage_bps,
            }
            row["estimated_transaction_cost"] = row["notional"] * cfg.transaction_cost_bps / 10_000.0
            row["estimated_slippage_cost"] = row["notional"] * cfg.slippage_bps / 10_000.0
            row["estimated_total_cost"] = row["estimated_transaction_cost"] + row["estimated_slippage_cost"]
            risk = risk_by_key.get((order.ticker, order.side, int(order.quantity)), {})
            row["risk_ok"] = bool(risk.get("ok", True))
            row["risk_reason"] = risk.get("reason", "")
            return row

        if cfg.dry_run:
            rows.extend(base_row(order) for order in safe_orders)
        elif cfg.submit_mode == "burst" and hasattr(broker, "place_orders_burst"):
            burst_rows = broker.place_orders_burst(
                safe_orders,
                pause_seconds=cfg.submit_pause_seconds,
                wait_seconds=cfg.post_submit_wait_seconds,
            )
            for order, burst in zip(safe_orders, burst_rows, strict=False):
                row = base_row(order)
                row.update(burst)
                row["notional"] = abs(row["quantity"] * row["limit_price"])
                row["estimated_transaction_cost"] = row["notional"] * cfg.transaction_cost_bps / 10_000.0
                row["estimated_slippage_cost"] = row["notional"] * cfg.slippage_bps / 10_000.0
                row["estimated_total_cost"] = row["estimated_transaction_cost"] + row["estimated_slippage_cost"]
                rows.append(row)
        elif cfg.submit_mode == "sequential":
            for order in safe_orders:
                row = base_row(order)
                try:
                    row["order_id"] = broker.place_order(order)
                except Exception as exc:
                    row["order_status"] = "rejected"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
        else:
            raise ValueError("submit_mode must be one of: burst, sequential")
    finally:
        if hasattr(broker, "ib"):
            broker.ib.disconnect()

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["ticker", "side", "quantity", "limit_price", "order_status", "order_id", "error"])
    out_path = report_path / "paper_order_results.csv"
    out.to_csv(out_path, index=False)
    return out
