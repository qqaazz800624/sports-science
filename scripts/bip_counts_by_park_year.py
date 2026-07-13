#%%

import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Count balls in play per park per year")
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                        help='Directory containing preprocessed MLB data')
    parser.add_argument('--input_filename', type=str, default='truncated_data_with_rtheta_team.parquet',
                        help='Filename for the truncated dataset with r_theta and team info')
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help='Directory to save the counts table')
    parser.add_argument('--output_filename', type=str, default='bip_counts_by_park_year.csv',
                        help='Filename to save the counts table')
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(args.data_dir, args.input_filename))

    # 與 prepare_regression_data 相同的樣本定義：例行賽打進場內的球
    df_bip = df[(df['description'] == 'hit_into_play') &
                (df['game_type'] == 'R')].copy()
    df_bip['home_team'] = df_bip['home_team'].replace({'ATH': 'OAK'})

    tbl = df_bip.groupby(['home_team', 'game_year']).size().unstack()
    tbl.index.name = 'Park'

    output_path = os.path.join(args.output_dir, args.output_filename)
    tbl.to_csv(output_path)
    print(f"Saved BIP counts to {output_path}")
    print(tbl.to_string())
    print()
    print(tbl.agg(['mean', 'min', 'max']).round(0).astype(int).to_string())


if __name__ == "__main__":
    main()

#%%
