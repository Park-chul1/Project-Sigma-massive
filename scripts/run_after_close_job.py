from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.daily_jobs import run_after_close_job
from live.paper_order_submitter import PaperSubmitConfig, submit_report_to_ibkr_paper


def main() -> None:
    p = argparse.ArgumentParser(description="Run after-close daily factor signal generation")
    p.add_argument("--date", required=True, help="Signal date YYYY-MM-DD")
    p.add_argument("--config", default=None)
    p.add_argument("--submit-paper-orders", action="store_true", help="Submit generated target portfolio to IBKR paper after report creation")
    p.add_argument("--paper-host", default="172.30.1.41")
    p.add_argument("--paper-port", type=int, default=7497)
    p.add_argument("--paper-client-id", type=int, default=72)
    p.add_argument("--paper-account", default=None)
    p.add_argument("--paper-dry-run", action="store_true", help="Build paper orders but do not call IBKR placeOrder")
    p.add_argument("--max-paper-orders", type=int, default=None)
    p.add_argument("--paper-transaction-cost-bps", type=float, default=1.0)
    p.add_argument("--paper-slippage-bps", type=float, default=2.0)
    p.add_argument("--paper-min-net-alpha-after-cost-bps", type=float, default=5.0)
    p.add_argument("--paper-min-price", type=float, default=0.0, help="Drop target positions whose order price is below this threshold; exits are still allowed")
    p.add_argument("--paper-submit-mode", choices=["burst", "sequential"], default="burst")
    p.add_argument("--paper-submit-pause-seconds", type=float, default=0.05)
    p.add_argument("--paper-post-submit-wait-seconds", type=float, default=5.0)
    args = p.parse_args()
    report_dir = run_after_close_job(args.date, args.config)
    print(f"after-close report: {report_dir}")
    if args.submit_paper_orders:
        result = submit_report_to_ibkr_paper(
            report_dir,
            PaperSubmitConfig(
                host=args.paper_host,
                port=args.paper_port,
                client_id=args.paper_client_id,
                account=args.paper_account,
                dry_run=args.paper_dry_run,
                max_orders=args.max_paper_orders,
                transaction_cost_bps=args.paper_transaction_cost_bps,
                slippage_bps=args.paper_slippage_bps,
                min_net_alpha_after_cost_bps=args.paper_min_net_alpha_after_cost_bps,
                min_price=args.paper_min_price,
                submit_mode=args.paper_submit_mode,
                submit_pause_seconds=args.paper_submit_pause_seconds,
                post_submit_wait_seconds=args.paper_post_submit_wait_seconds,
            ),
        )
        print(f"paper order results: {report_dir / 'paper_order_results.csv'}")
        print(result[["ticker", "side", "quantity", "limit_price", "order_status", "order_id", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
