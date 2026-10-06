from __future__ import annotations

from dataclasses import dataclass
import itertools
import logging
import math
import time
from typing import Protocol

import pandas as pd

from live.config import LiveConfig


@dataclass(frozen=True)
class Order:
    ticker: str
    side: str
    quantity: int
    limit_price: float
    dry_run: bool = False


class BrokerClient(Protocol):
    def get_positions(self) -> pd.Series: ...
    def get_account_summary(self) -> dict: ...
    def place_order(self, order: Order) -> str: ...
    def cancel_order(self, order_id: str) -> None: ...
    def get_open_orders(self) -> list[Order]: ...


class SimulatedPaperBroker:
    def __init__(self, config: LiveConfig):
        self.config = config
        self.cash = float(config.PAPER_STARTING_EQUITY)
        self.positions: dict[str, int] = {}
        self.open_orders: dict[str, Order] = {}
        self._ids = itertools.count(1)

    def get_positions(self) -> pd.Series:
        return pd.Series(self.positions, dtype=float)

    def get_account_summary(self) -> dict:
        return {"equity": self.config.PAPER_STARTING_EQUITY, "cash": self.cash, "paper": True}

    def place_order(self, order: Order) -> str:
        oid = f"PAPER-{next(self._ids)}"
        logging.info("paper order %s %s %s %s @ %.4f", oid, order.side, order.quantity, order.ticker, order.limit_price)
        if not order.dry_run:
            signed = order.quantity if order.side.upper() == "BUY" else -order.quantity
            self.positions[order.ticker] = self.positions.get(order.ticker, 0) + signed
        return oid

    def cancel_order(self, order_id: str) -> None:
        self.open_orders.pop(order_id, None)

    def get_open_orders(self) -> list[Order]:
        return list(self.open_orders.values())


def make_broker(config: LiveConfig) -> BrokerClient:
    if config.PAPER_TRADING:
        if config.BROKER == "ibkr_paper":
            try:
                return IBKRPaperBroker(config)
            except ImportError:
                logging.warning("ib_insync is not installed; falling back to simulated paper broker")
        return SimulatedPaperBroker(config)
    if not config.ENABLE_REAL_TRADING:
        raise RuntimeError("Real broker requested but ENABLE_REAL_TRADING is false")
    raise NotImplementedError("Real broker adapters are intentionally not enabled by default")


class IBKRPaperBroker:
    def __init__(self, config: LiveConfig):
        if not config.PAPER_TRADING or config.ENABLE_REAL_TRADING:
            raise RuntimeError("IBKRPaperBroker requires PAPER_TRADING=true and ENABLE_REAL_TRADING=false")
        from ib_insync import IB

        self.config = config
        self.ib = IB()
        self.ib.connect(config.IBKR_HOST, config.IBKR_PORT, clientId=config.IBKR_BROKER_CLIENT_ID, timeout=10)

    def _stock(self, ticker: str):
        from ib_insync import Stock

        if "_" in ticker:
            raise ValueError("Broker orders require a resolved trading symbol, not a Compustat security identifier")
        return Stock(ticker, "SMART", "USD")

    def _min_tick(self, contract) -> float:
        try:
            details = self.ib.reqContractDetails(contract)
            ticks = [float(d.minTick) for d in details if getattr(d, "minTick", None)]
            ticks = [t for t in ticks if t > 0]
            if ticks:
                return min(ticks)
        except Exception as exc:
            logging.warning("IBKR minTick lookup failed ticker=%s error=%s", getattr(contract, "symbol", ""), exc)
        return 0.01

    def _round_limit_to_tick(self, side: str, price: float, min_tick: float) -> float:
        if price >= 1.0:
            min_tick = max(min_tick, 0.01)
        if min_tick <= 0:
            return round(price, 4)
        steps = price / min_tick
        if side.upper() == "BUY":
            rounded = math.ceil(steps - 1e-12) * min_tick
        else:
            rounded = math.floor(steps + 1e-12) * min_tick
        decimals = max(0, min(6, int(math.ceil(-math.log10(min_tick))) if min_tick < 1 else 0))
        return round(rounded, decimals)

    def get_positions(self) -> pd.Series:
        positions: dict[str, float] = {}
        for pos in self.ib.positions():
            if self.config.IBKR_ACCOUNT and pos.account != self.config.IBKR_ACCOUNT:
                continue
            symbol = getattr(pos.contract, "symbol", None)
            if symbol:
                positions[symbol] = float(pos.position)
        return pd.Series(positions, dtype=float)

    def get_account_summary(self) -> dict:
        out = {"paper": True}
        for value in self.ib.accountValues():
            if self.config.IBKR_ACCOUNT and value.account != self.config.IBKR_ACCOUNT:
                continue
            if value.tag == "EquityWithLoanValue":
                out["equity"] = float(value.value)
            elif value.tag == "CashBalance":
                out["cash"] = float(value.value)
        out.setdefault("equity", self.config.PAPER_STARTING_EQUITY)
        return out

    def place_order(self, order: Order) -> str:
        from ib_insync import LimitOrder

        contracts = self.ib.qualifyContracts(self._stock(order.ticker))
        if not contracts:
            raise RuntimeError(f"IBKR could not qualify contract for {order.ticker}")
        contract = contracts[0]
        min_tick = self._min_tick(contract)
        limit_price = self._round_limit_to_tick(order.side, order.limit_price, min_tick)
        ib_order = LimitOrder(
            order.side.upper(),
            order.quantity,
            limit_price,
            tif="DAY",
            account=self.config.IBKR_ACCOUNT or "",
        )
        ib_order.overridePercentageConstraints = True
        trade = self.ib.placeOrder(contracts[0], ib_order)
        for _ in range(20):
            self.ib.sleep(0.25)
            if trade.orderStatus.status in {"Submitted", "PreSubmitted", "Filled", "Cancelled", "Inactive"}:
                break
        logging.info(
            "IBKR paper order id=%s ticker=%s side=%s qty=%s limit=%.4f status=%s filled=%s remaining=%s",
            trade.order.orderId,
            order.ticker,
            order.side,
            order.quantity,
            limit_price,
            trade.orderStatus.status,
            trade.orderStatus.filled,
            trade.orderStatus.remaining,
        )
        print(
            f"IBKR paper order id={trade.order.orderId} ticker={order.ticker} side={order.side} "
            f"qty={order.quantity} limit={limit_price:.6g} status={trade.orderStatus.status} "
            f"filled={trade.orderStatus.filled} remaining={trade.orderStatus.remaining}",
            flush=True,
        )
        if trade.orderStatus.status in {"Cancelled", "Inactive"}:
            last_log = trade.log[-1].message if trade.log else ""
            raise RuntimeError(f"IBKR paper order {trade.order.orderId} {order.ticker} status={trade.orderStatus.status}: {last_log}")
        return str(trade.order.orderId)

    def place_orders_burst(
        self,
        orders: list[Order],
        pause_seconds: float = 0.05,
        wait_seconds: float = 5.0,
    ) -> list[dict]:
        from ib_insync import LimitOrder

        results: list[dict] = []
        trades = []
        for order in orders:
            result = {
                "ticker": order.ticker,
                "side": order.side,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
                "order_status": "submitted",
                "order_id": "",
                "error": "",
            }
            try:
                contracts = self.ib.qualifyContracts(self._stock(order.ticker))
                if not contracts:
                    raise RuntimeError(f"IBKR could not qualify contract for {order.ticker}")
                contract = contracts[0]
                min_tick = self._min_tick(contract)
                limit_price = self._round_limit_to_tick(order.side, order.limit_price, min_tick)
                ib_order = LimitOrder(
                    order.side.upper(),
                    order.quantity,
                    limit_price,
                    tif="DAY",
                    account=self.config.IBKR_ACCOUNT or "",
                )
                ib_order.overridePercentageConstraints = True
                trade = self.ib.placeOrder(contract, ib_order)
                result["limit_price"] = limit_price
                result["order_id"] = str(trade.order.orderId)
                trades.append((order, trade, result))
                print(
                    f"IBKR paper submitted id={trade.order.orderId} ticker={order.ticker} "
                    f"side={order.side} qty={order.quantity} limit={limit_price:.6g}",
                    flush=True,
                )
                if pause_seconds > 0:
                    self.ib.sleep(pause_seconds)
            except Exception as exc:
                result["order_status"] = "rejected"
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)

        deadline = time.monotonic() + max(wait_seconds, 0.0)
        terminal = {"Submitted", "PreSubmitted", "Filled", "Cancelled", "Inactive"}
        while time.monotonic() < deadline:
            if all(trade.orderStatus.status in terminal for _, trade, _ in trades):
                break
            self.ib.sleep(0.25)

        for order, trade, result in trades:
            status = trade.orderStatus.status or "submitted"
            result["order_status"] = status
            result["order_id"] = str(trade.order.orderId)
            logging.info(
                "IBKR paper burst id=%s ticker=%s side=%s qty=%s limit=%s status=%s filled=%s remaining=%s",
                trade.order.orderId,
                order.ticker,
                order.side,
                order.quantity,
                result["limit_price"],
                status,
                trade.orderStatus.filled,
                trade.orderStatus.remaining,
            )
            if status in {"Cancelled", "Inactive"}:
                last_log = trade.log[-1].message if trade.log else ""
                result["error"] = last_log
        return results

    def cancel_order(self, order_id: str) -> None:
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == str(order_id):
                self.ib.cancelOrder(trade.order)

    def get_open_orders(self) -> list[Order]:
        orders: list[Order] = []
        for trade in self.ib.openTrades():
            contract = trade.contract
            order = trade.order
            orders.append(Order(
                ticker=getattr(contract, "symbol", ""),
                side=str(order.action),
                quantity=int(order.totalQuantity),
                limit_price=float(order.lmtPrice),
            ))
        return orders
