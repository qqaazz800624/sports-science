#!/bin/bash

python scripts/estimate_factors.py \
    --data_dir "/neodata/open_dataset/mlb_data/preprocessed" \
    --input_filename "truncated_data_with_rtheta_team.parquet" \
    --prob_table "rtheta_prob_tbl.parquet" \
    --output_dir "/neodata/open_dataset/mlb_data/results" \
    --output_filename "estimated_factors.csv" \
    --weights 1.0 2.0 3.0 4.0