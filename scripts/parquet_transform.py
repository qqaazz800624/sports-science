#%%

import pandas as pd
import os
from tqdm import tqdm
import argparse
from utils import add_rtheta_features, assign_pitcher_batter_teams

#%%

def main():
    parser = argparse.ArgumentParser(description="Merge MLB Statcast Data")
    parser.add_argument('--data_dir', type=str, default='/neodata/open_dataset/mlb_data',
                        help='Directory containing MLB Statcast CSV files')
    parser.add_argument('--save_dir', type=str, default='/neodata/open_dataset/mlb_data/preprocessed',
                        help='Directory to save the merged Parquet file')
    parser.add_argument('--parquet_filename', type=str, default='savant_data_14_24.parquet',
                        help='Filename for the merged Parquet file')
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    parquet_filename = args.parquet_filename

    all_data = []

    for file in tqdm(os.listdir(data_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            year = file.split('_')[-1].split('.')[0]
            df['year'] = year 
            all_data.append(df)
        else:
            continue
    
    merged_df = pd.concat(all_data, ignore_index=True)
    columns_filtered = ['pitch_type', 'game_type', 'game_pk',
                        'game_date', 'game_year', 'player_name',
                        'pitch_name', 'post_bat_score', 'bat_score',
                        'batter', 'pitcher', 'events', 'description', 
                        'inning_topbot', 'home_team', 'away_team',
                        'launch_speed', 'launch_angle', 'bb_type',
                        'hc_x', 'hc_y', 'hit_location']
    merged_df = merged_df[columns_filtered]
    merged_df = merged_df[merged_df['game_type'] == 'R'] # added filter for regular season games
    add_rtheta_df = add_rtheta_features(merged_df)
    df_teams = assign_pitcher_batter_teams(add_rtheta_df)
    save_path = os.path.join(save_dir, parquet_filename)
    df_teams.to_parquet(save_path)

if __name__ == "__main__":
    main()


#%%

