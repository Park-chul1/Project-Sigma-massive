from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import json
import os


@dataclass(frozen=True)
class LiveConfig:
    DATA_PROVIDER: str = "wrds"
    DELAY_MINUTES: int = 0
    BAR_INTERVAL: str = "1d"
    POLL_INTERVAL_SECONDS: int = 86400
    UNIVERSE_SIZE_LIMIT: int | None = None
    MIN_DOLLAR_VOLUME: float = 1_000_000.0
    MAX_POSITION_WEIGHT: float = 0.02
    MAX_GROSS_EXPOSURE: float = 1.0
    MAX_NET_EXPOSURE: float = 0.10
    MAX_TURNOVER_PER_REBALANCE: float = 0.10
    MIN_HOLDING_PERIOD_BARS: int = 4
    ALPHA_Z_THRESHOLD: float = 1.5
    COST_BPS: float = 1.0
    SLIPPAGE_BPS: float = 2.0
    MIN_NET_ALPHA_AFTER_COST_BPS: float = 5.0
    PAPER_TRADING: bool = True
    ENABLE_REAL_TRADING: bool = False
    BROKER: str = "sim_paper"
    IBKR_HOST: str = "172.30.1.41"
    IBKR_PORT: int = 7497
    IBKR_CLIENT_ID: int = 71
    IBKR_BROKER_CLIENT_ID: int = 72
    IBKR_ACCOUNT: str | None = None
    WRDS_USERNAME: str | None = None
    WRDS_CACHE_PATH: Path = Path("data/wrds_api_cache")
    WRDS_PREFETCH_DAYS: int = 400
    MEMBERSHIP_PATH: Path = Path("data/processed_wrds/nasdaq_membership.parquet")
    LONG_SHORT: bool = True
    DRY_RUN: bool = False
    LOOKBACK_BARS: int = 260
    MIN_FACTOR_NAMES: int = 10
    MIN_EXPOSURE_COVERAGE: float = 0.70
    MIN_ORDER_DOLLARS: float = 100.0
    MARKET_TIMEZONE: str = "America/New_York"
    CACHE_PATH: Path = Path("data/live_wrds/bars.parquet")
    UNIVERSE_PATH: Path = Path("data/processed_wrds/tickers.csv")
    UNIVERSE_BARS_PATH: Path = Path("data/processed_wrds/daily_bars.parquet")
    FACTOR_PRED_PATH: Path = Path("data/processed_wrds/factor_returns.npy")
    FACTOR_NAMES_PATH: Path = Path("data/processed_wrds/factor_names.csv")
    LOG_DIR: Path = Path("logs")
    PAPER_STARTING_EQUITY: float = 100_000.0

    def validate(self) -> None:
        if self.DATA_PROVIDER.lower() not in {"wrds", "cache"}:
            raise ValueError("DATA_PROVIDER must be wrds or cache; all market data must originate in Compustat WRDS")
        if self.BAR_INTERVAL != "1d":
            raise ValueError("Compustat WRDS supplies daily bars only; intraday market data is unsupported")
        if self.DELAY_MINUTES != 0:
            raise ValueError("Minute-delay settings are unsupported for Compustat WRDS daily data")
        if self.PAPER_TRADING and self.ENABLE_REAL_TRADING:
            raise ValueError("PAPER_TRADING and ENABLE_REAL_TRADING cannot both be true")
        if not self.PAPER_TRADING and not self.ENABLE_REAL_TRADING:
            raise RuntimeError("Refusing to run: PAPER_TRADING is false and real trading is not explicitly enabled")
        if self.ENABLE_REAL_TRADING and os.getenv("LIVE_TRADING_CONFIRM") != "I_UNDERSTAND_REAL_MONEY_RISK":
            raise RuntimeError("Real trading requires LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY_RISK")
        if self.MAX_POSITION_WEIGHT <= 0:
            raise ValueError("MAX_POSITION_WEIGHT must be positive")
        if self.MAX_GROSS_EXPOSURE <= 0:
            raise ValueError("MAX_GROSS_EXPOSURE must be positive")
        if self.MAX_TURNOVER_PER_REBALANCE < 0:
            raise ValueError("MAX_TURNOVER_PER_REBALANCE cannot be negative")
        if self.COST_BPS < 0 or self.SLIPPAGE_BPS < 0 or self.MIN_NET_ALPHA_AFTER_COST_BPS < 0:
            raise ValueError("cost, slippage, and net alpha thresholds must be non-negative")


def load_config(path: str | Path | None = None) -> LiveConfig:
    if path is None:
        cfg = LiveConfig()
        cfg.validate()
        return cfg

    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    names = {f.name for f in fields(LiveConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"Unknown live config keys: {sorted(unknown)}")
    path_fields = {"CACHE_PATH", "UNIVERSE_PATH", "UNIVERSE_BARS_PATH", "FACTOR_PRED_PATH", "FACTOR_NAMES_PATH", "LOG_DIR", "WRDS_CACHE_PATH", "MEMBERSHIP_PATH"}
    for name in path_fields & set(raw):
        raw[name] = Path(raw[name])
    cfg = LiveConfig(**raw)
    cfg.validate()
    return cfg
