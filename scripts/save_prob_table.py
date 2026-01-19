#%%

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

#%%

def main():
    parser = argparse.ArgumentParser(description="Save R-Theta Probability Table")
    parser.add_argument('--data_dir', type=str, default='/neodata/open_dataset/mlb_data/preprocessed',
                        help='Directory containing preprocessed MLB Statcast Parquet file')
    parser.add_argument('--input_filename', type=str, default='truncated_data_with_rtheta_team.parquet',
                        help='Input Parquet filename with r_theta and team info')
    parser.add_argument('--output_filename', type=str, default='rtheta_prob_tbl.parquet',
                        help='Output Parquet filename for r_theta probability table')
    args = parser.parse_args()

    data_dir = args.data_dir
    input_filename = args.input_filename
    output_filename = args.output_filename
    file_path = os.path.join(data_dir, input_filename)
    df = pd.read_parquet(file_path)

    rtheta_prob_table = df.groupby('r_theta')['events'].value_counts(normalize=True).reset_index(name='probability')

    save_path = os.path.join(data_dir, output_filename)
    rtheta_prob_table.to_parquet(save_path)

if __name__ == "__main__":
    main()

#%%


