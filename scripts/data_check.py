#%%

import pandas as pd
import numpy as np
import os
from utils import get_expected_bases_map, Config
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


#%%

df_regular = df[df['game_type'] == 'R'].copy()

team_mapping = {'ATH': 'OAK'}
df_regular['home_team'] = df_regular['home_team'].replace(team_mapping)
df_regular['away_team'] = df_regular['away_team'].replace(team_mapping)

unique_games = df_regular.drop_duplicates(subset=['game_pk']).copy()

home_counts = unique_games.groupby(['game_year', 'home_team']).size().reset_index(name='home_games')
home_counts.rename(columns={'home_team': 'team'}, inplace=True)

away_counts = unique_games.groupby(['game_year', 'away_team']).size().reset_index(name='away_games')
away_counts.rename(columns={'away_team': 'team'}, inplace=True)

team_games = pd.merge(home_counts, away_counts, on=['game_year', 'team'], how='outer').fillna(0)

team_games['total_games'] = team_games['home_games'] + team_games['away_games']

game_matrix = team_games.pivot(index='team', columns='game_year', values='total_games').fillna(0).astype(int)

# 儲存結果並印出
#game_matrix.to_csv('team_games_per_year.csv')
print("Team Games Per Year Matrix:")
print(game_matrix)


#%%

data_dir = '/neodata/open_dataset/mlb_data'
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

#%%

df_regular = merged_df[merged_df['game_type'] == 'R'].copy()

team_mapping = {'ATH': 'OAK'}
df_regular['home_team'] = df_regular['home_team'].replace(team_mapping)
df_regular['away_team'] = df_regular['away_team'].replace(team_mapping)

unique_games = df_regular.drop_duplicates(subset=['game_pk']).copy()

home_counts = unique_games.groupby(['game_year', 'home_team']).size().reset_index(name='home_games')
home_counts.rename(columns={'home_team': 'team'}, inplace=True)

away_counts = unique_games.groupby(['game_year', 'away_team']).size().reset_index(name='away_games')
away_counts.rename(columns={'away_team': 'team'}, inplace=True)

team_games = pd.merge(home_counts, away_counts, on=['game_year', 'team'], how='outer').fillna(0)

team_games['total_games'] = team_games['home_games'] + team_games['away_games']

game_matrix = team_games.pivot(index='team', columns='game_year', values='total_games').fillna(0).astype(int)

# 儲存結果並印出
game_matrix.to_csv('team_games_per_year.csv')
print("各球隊每年出賽場次表 (以 game_pk 統計)：")
print(game_matrix)

#%%

data_path = "/neodata/open_dataset/mlb_data/statcast_2024.csv"
df_2024 = pd.read_csv(data_path)

df_2024_regular = df_2024[df_2024['game_type'] == 'R']
stl_df = df_2024_regular[(df_2024_regular['home_team'] == 'STL') | (df_2024_regular['away_team'] == 'STL')]

stl_game_pks = stl_df['game_pk'].unique()
print(f"STL 2024 Regular Season Games: {len(stl_game_pks)}")



#%%











#%%







#%%








#%%