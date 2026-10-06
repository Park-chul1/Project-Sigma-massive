"""Warm the Compustat daily cache; daily signal/execution jobs own trading."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.config import load_config
from live.data import load_universe, make_provider, update_local_bar_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch Compustat WRDS daily bars for the NASDAQ universe")
    parser.add_argument("--config", default=None, help="Optional WRDS live JSON configuration")
    parser.add_argument("--prefetch-history", action="store_true", help="Fetch historical daily bars")
    parser.add_argument("--prefetch-only", action="store_true", help="Fetch bars without generating signals or orders")
    parser.add_argument("--prefetch-limit", type=int, default=None)
    parser.add_argument("--asof", default=None, help="Latest completed trading date (YYYY-MM-DD)")
    args = parser.parse_args()
    if not args.prefetch_only:
        parser.error(
            "Compustat WRDS is daily only. Use --prefetch-only for cache updates, "
            "scripts/run_after_close_job.py for signals, and scripts/submit_daily_paper_orders.py for execution."
        )
    config = load_config(args.config)
    universe = load_universe(config, args.asof)
    provider = make_provider(config)
    if not hasattr(provider, "prefetch_history"):
        parser.error("Prefetch requires DATA_PROVIDER=wrds")
    try:
        result = provider.prefetch_history(universe, asof_time=args.asof, limit=args.prefetch_limit)
        update_local_bar_cache(result.bars, config.CACHE_PATH)
    finally:
        if hasattr(provider, "disconnect"):
            provider.disconnect()
    print(f"WRDS daily cache updated through {result.bars['timestamp'].max().date()}: {len(result.bars)} rows, {result.symbols_updated} securities")


if __name__ == "__main__":
    main()
