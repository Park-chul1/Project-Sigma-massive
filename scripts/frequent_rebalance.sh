#!/bin/bash
cd /home/chul/projects/Quant2/factor_pipeline_clean
source .venv/bin/activate

echo "$(date): Starting frequent portfolio rebalancing..." >> logs/frequent_rebalance.log

# Run portfolio rebalancing only (factor pipeline runs daily)
echo "$(date): Running portfolio rebalancing..." >> logs/frequent_rebalance.log
python scripts/ibkr_live_portfolio.py --load-pipeline data/processed --host 172.30.1.41 --port 7497 --auto-rebalance --date-index -1 --method ewma --lookback 20 --ewma-halflife 20.0 --quantile 0.10 --gross 2.0 --notional 3000.0 --top-n 15 --max-symbol-pct 0.03 --max-side-pct 0.90 >> logs/frequent_rebalance.log 2>&1

if [ $? -eq 0 ]; then
    echo "$(date): Portfolio rebalancing completed successfully" >> logs/frequent_rebalance.log
else
    echo "$(date): Portfolio rebalancing failed" >> logs/frequent_rebalance.log
fi

echo "$(date): Frequent rebalancing finished" >> logs/frequent_rebalance.log