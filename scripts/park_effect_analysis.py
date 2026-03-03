#%%

import pandas as pd
import numpy as np
import os
from utils import get_expected_bases_map, Config
import argparse
from tqdm import tqdm

current_weights = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'home_run': 4
}

data_dir = "/neodata/open_dataset/mlb_data/preprocessed"
prob_table = "rtheta_prob_tbl.parquet"
input_filename = "truncated_data_with_rtheta_team.parquet"
truncated_file_path = os.path.join(data_dir, input_filename)

config = Config(
    weights=current_weights,
    data_dir=data_dir,
    filename=prob_table
)

exp_map = get_expected_bases_map(config=config)
df = pd.read_parquet(truncated_file_path)
df_bip = df[
            (df['description'] == 'hit_into_play') & 
            (df['game_type'] == 'R')].copy()

team_mapping = {'ATH': 'OAK'}
df_bip['home_team'] = df_bip['home_team'].replace(team_mapping)
df_bip['pitcher_team'] = df_bip['pitcher_team'].replace(team_mapping)
df_bip['batter_team'] = df_bip['batter_team'].replace(team_mapping)
df_bip['expected_metric'] = df_bip['r_theta'].map(exp_map).fillna(0)
event_weights = config.weights
df_bip['real_metric'] = df_bip['events'].map(event_weights).fillna(0)
df_bip['YN_BOS'] = df_bip['home_team'].apply(lambda x: 1 if x == 'BOS' else 0)

speed_bins = range(0, 121, 10)      # 0, 10, 20, ..., 120
angle_bins = range(-90, 91, 10)     # -90, -80, ..., 90

speed_labels = [
    f"{i:02d}. [{low}, {high})" 
    for i, (low, high) in enumerate(zip(speed_bins[:-1], speed_bins[1:]), 1)
]

angle_labels = [
    f"{i:02d}. [{low}, {high})" 
    for i, (low, high) in enumerate(zip(angle_bins[:-1], angle_bins[1:]), 1)
]

df_bip['speed_bin'] = pd.cut(df_bip['launch_speed'], bins=speed_bins, right=False, labels=speed_labels)
df_bip['angle_bin'] = pd.cut(df_bip['launch_angle'], bins=angle_bins, right=False, labels=angle_labels)

df_bip['expected_metric_raw'] = df_bip['r_theta'].map(exp_map)
valid_df = df_bip.dropna(subset=['expected_metric_raw']).copy()
valid_df['expected_metric'] = valid_df['expected_metric_raw'].fillna(0) 
valid_df['residual'] = valid_df['real_metric'] - valid_df['expected_metric']

#%%

valid_df['is_batting_home'] = (valid_df['batter_team'] == valid_df['home_team'])

batting_stats = valid_df.groupby(['game_year', 'batter_team', 'is_batting_home'])['residual'].mean().unstack()
#batting_stats = -batting_stats

#batting_stats.columns = ['batting_away_residual', 'batting_home_residual']
#batting_stats.columns = ['opp_defense_away_residual', 'opp_defense_home_residual']
batting_stats.columns = ['Opp Defense TBR (away)', 'Opp Defense TBR (home)']
batting_stats.reset_index(inplace=True)


valid_df['is_pitching_home'] = (valid_df['pitcher_team'] == valid_df['home_team'])
pitching_stats = valid_df.groupby(['game_year', 'pitcher_team', 'is_pitching_home'])['residual'].mean().unstack()
#pitching_stats = -pitching_stats

#pitching_stats.columns = ['pitching_away_residual', 'pitching_home_residual']
#pitching_stats.columns = ['team_defense_away_residual', 'team_defense_home_residual']
pitching_stats.columns = ['Team Defense TBR (away)', 'Team Defense TBR (home)']
pitching_stats.reset_index(inplace=True)

final_stats = pd.merge(
    batting_stats,
    pitching_stats,
    left_on=['game_year', 'batter_team'],
    right_on=['game_year', 'pitcher_team'],
    how='outer'
)

final_stats.rename(columns={'batter_team': 'team'}, inplace=True)
final_stats.drop(columns=['pitcher_team'], inplace=True)

final_stats.to_csv('appendix_c_TBR.csv', index=False)

#%%

group_cols = ['game_year', 'speed_bin', 'angle_bin', 'YN_BOS']

agg_df = df_bip.groupby(group_cols).agg({
        'real_metric': 'sum',
        'expected_metric': 'sum',
        'events': 'count' 
}).reset_index()

#%%

diff_sum = agg_df['real_metric'] - agg_df['expected_metric']

agg_df['diff_metric'] = np.where(
    agg_df['events'] > 0, 
    diff_sum / agg_df['events'], 
    0  
)

#%%


agg_df.head(10)


#%%

agg_df.to_csv('agg_by_r_theta_bin_bos.csv', index=False)





#%%






#%%








#%%






#%%








#%%






#%%
