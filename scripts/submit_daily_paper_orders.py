from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.paper_order_submitter import PaperSubmitConfig, submit_report_to_ibkr_paper


def main() -> None:
    p = argparse.ArgumentParser(description="Submit a daily report target portfolio to IBKR paper")
    p.add_argument("--report-dir", required=True, help="reports/daily/YYYY-MM-DD directory")
    p.add_argument("--host", default="172.30.1.41")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=72)
    p.add_argument("--account", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-orders", type=int, default=None)
    p.add_argument("--transaction-cost-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--min-net-alpha-after-cost-bps", type=float, default=5.0)
    p.add_argument("--min-price", type=float, default=0.0, help="Drop target positions whose order price is below this threshold; exits are still allowed")
    p.add_argument("--submit-mode", choices=["burst", "sequential"], default="burst")
    p.add_argument("--submit-pause-seconds", type=float, default=0.05)
    p.add_argument("--post-submit-wait-seconds", type=float, default=5.0)
    args = p.parse_args()
    result = submit_report_to_ibkr_paper(
        Path(args.report_dir),
        PaperSubmitConfig(
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            account=args.account,
            dry_run=args.dry_run,
            max_orders=args.max_orders,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            min_net_alpha_after_cost_bps=args.min_net_alpha_after_cost_bps,
            min_price=args.min_price,
            submit_mode=args.submit_mode,
            submit_pause_seconds=args.submit_pause_seconds,
            post_submit_wait_seconds=args.post_submit_wait_seconds,
        ),
    )
    print(f"paper order results: {Path(args.report_dir) / 'paper_order_results.csv'}")
    print(result[["ticker", "side", "quantity", "limit_price", "order_status", "order_id", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
