#!/bin/bash

python scripts/parquet_transform.py \
    --data_dir "/Users/wujhejia/Documents/sports-science/data" \
    --save_dir "/Users/wujhejia/Documents/sports-science/data/preprocessed" \
    --parquet_filename "truncated_data_with_rtheta_team.parquet"
