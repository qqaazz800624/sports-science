#!/bin/bash

python scripts/save_prob_table.py \
    --data_dir "/Users/wujhejia/Documents/sports-science/data/preprocessed" \
    --input_filename "truncated_data_with_rtheta_team.parquet" \
    --output_filename "rtheta_prob_tbl.parquet"
    