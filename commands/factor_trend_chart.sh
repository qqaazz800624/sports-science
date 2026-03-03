#!/bin/bash

export factor="ParkFactor"

python scripts/factor_trend_chart.py \
    --factor "$factor" \
    --teams "COL" "CIN" "MIN" "MIL" "PIT" \
    --data_dir "/neodata/open_dataset/mlb_data" \
    --input_filename "results/estimated_factors.csv" \
    --save_dir "/neodata/open_dataset/mlb_data/results" \
    --save_filename "${factor}_trend_chart.png"