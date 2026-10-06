#!/bin/bash
cd /home/chul/projects/Quant2/factor_pipeline_clean
source .venv/bin/activate

echo "$(date): Starting automated pipeline..." >> logs/auto_pipeline.log

# Run factor pipeline
echo "$(date): Running factor pipeline..." >> logs/auto_pipeline.log
python scripts/run_dynamic_factor_pipeline.py --output-dir data/processed --q-scale 1e-4 --r-scale 1e-2 --lookback 20 --warning-q 0.95 --break-q 0.99 >> logs/auto_pipeline.log 2>&1

if [ $? -eq 0 ]; then
    echo "$(date): Factor pipeline completed successfully" >> logs/auto_pipeline.log
    
    # Run portfolio rebalancing
    echo "$(date): Running portfolio rebalancing..." >> logs/auto_pipeline.log
    python scripts/ibkr_live_portfolio.py --load-pipeline data/processed --host 172.30.1.41 --port 7497 --auto-rebalance --date-index -1 --method ewma --lookback 20 --ewma-halflife 20.0 --quantile 0.10 --gross 2.0 --notional 5000.0 --top-n 10 --max-symbol-pct 0.05 --max-side-pct 0.80 >> logs/auto_pipeline.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "$(date): Portfolio rebalancing completed successfully" >> logs/auto_pipeline.log
    else
        echo "$(date): Portfolio rebalancing failed" >> logs/auto_pipeline.log
    fi
else
    echo "$(date): Factor pipeline failed" >> logs/auto_pipeline.log
fi

echo "$(date): Automated pipeline finished" >> logs/auto_pipeline.log
