#!/bin/bash

python scripts/parquet_transform.py \
    --data_dir "/neodata/open_dataset/mlb_data" \
    --save_dir "/neodata/open_dataset/mlb_data/preprocessed" \
    --parquet_filename "truncated_data_with_rtheta_team.parquet"
