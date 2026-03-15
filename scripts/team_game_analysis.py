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
df_bip['away_team'] = df_bip['away_team'].replace(team_mapping)
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

target_team = 'AZ'
target_year = 2024


target_df = valid_df[valid_df['game_year'] == target_year].copy()
stl_df = target_df[(target_df['home_team'] == target_team) | (target_df['away_team'] == target_team)]

stl_games = stl_df.drop_duplicates(subset=['game_pk']).copy()

stl_games['venue'] = np.where(stl_games['home_team'] == target_team, 'Home', 'Away')
stl_games['opponent'] = np.where(stl_games['home_team'] == target_team, stl_games['away_team'], stl_games['home_team'])

game_counts = stl_games.groupby(['venue', 'opponent']).size().reset_index(name='game_count')

summary_table = game_counts.pivot_table(
    index='opponent', 
    columns='venue', 
    values='game_count', 
    aggfunc='sum', 
    fill_value=0
)

if 'Home' not in summary_table.columns: summary_table['Home'] = 0
if 'Away' not in summary_table.columns: summary_table['Away'] = 0

summary_table['Total'] = summary_table['Home'] + summary_table['Away']
#summary_table = summary_table.sort_values(by='Total', ascending=False)
summary_table = summary_table.sort_index()

print(f"{target_year} {target_team} 在 valid_df 中的總場次: {summary_table['Total'].sum()} 場")
print(summary_table)

summary_table.to_csv(f'{target_team}_game_summary_{target_year}.csv')

#%% Home Angle

target_year = 2024
home_team = 'AZ'
pitcher_team = 'AZ'

target_df = valid_df[(valid_df['game_year'] == target_year) &
                     (valid_df['home_team'] == home_team) &
                     (valid_df['pitcher_team'] == pitcher_team)].copy()
groupby_cols = ['batter_team']

summary_df = target_df.groupby(groupby_cols).agg(
    mean_residual=('residual', 'mean'),      
    games_played=('game_pk', 'nunique')      
).reset_index()

summary_df = summary_df.sort_index()

print(f"{target_year} {home_team} Home ({pitcher_team} Defense), Mean Total Bases Residual & Games Played by Opponent:")
print(summary_df)
summary_df.to_csv(f'{home_team}_home_{pitcher_team}_defense_summary_{target_year}.csv', index=False)

#%% Away Angle

target_year = 2024
away_team = 'AZ'
pitcher_team = 'AZ'

target_df = valid_df[(valid_df['game_year'] == target_year) &
                     (valid_df['away_team'] == away_team) &
                     (valid_df['pitcher_team'] == pitcher_team)].copy()

groupby_cols = ['batter_team']

summary_df = target_df.groupby(groupby_cols).agg(
    mean_residual=('residual', 'mean'),      
    games_played=('game_pk', 'nunique')      
).reset_index()

summary_df = summary_df.sort_index()

print(f"{target_year} {away_team} Away ({pitcher_team} Defense), Mean Total Bases Residual & Games Played by Opponent:")
print(summary_df)
summary_df.to_csv(f'{away_team}_away_{pitcher_team}_defense_summary_{target_year}.csv', index=False)

#%%

target_team = 'AZ'
target_year = 2024

target_df = valid_df[valid_df['game_year'] == target_year].copy()

stl_df = target_df[(target_df['home_team'] == target_team) | (target_df['away_team'] == target_team)].copy()


stl_df['venue'] = np.where(stl_df['home_team'] == target_team, 'Home', 'Away')
stl_df['opponent'] = np.where(stl_df['home_team'] == target_team, stl_df['away_team'], stl_df['home_team'])

bip_counts = stl_df.groupby(['venue', 'opponent']).size().reset_index(name='bip_count')

summary_table = bip_counts.pivot_table(
    index='opponent', 
    columns='venue', 
    values='bip_count', 
    aggfunc='sum', 
    fill_value=0
)

if 'Home' not in summary_table.columns: summary_table['Home'] = 0
if 'Away' not in summary_table.columns: summary_table['Away'] = 0

summary_table['Total'] = summary_table['Home'] + summary_table['Away']
summary_table = summary_table.sort_index()

print(f"{target_year} {target_team} 在 valid_df 中的總擊球入場次數 (BIP): {summary_table['Total'].sum()} 球")
print(summary_table)

summary_table.to_csv(f'{target_team}_bip_summary_{target_year}.csv')


#%%






#%%







#%%






#%%







#%%






#%%