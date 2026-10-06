from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    start: str = "2019-01-01"
    end: str = "2024-12-31"
    wrds_username: str | None = None
    out_dir: Path = Path("data/processed_wrds")
    cache_dir: Path = Path("data/cache_wrds")
    min_names: int = 30
    ridge: float = 1e-4
    forward_horizon: int = 1
    financial_timeframe: str = "quarterly"
    financial_lookback_days: int = 550
    financial_lag_days: int = 60
    ttm_min_quarters: int = 4
    max_factor_corr: float = 0.999
    corr_min_overlap: int = 100
    use_cache: bool = True
