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
save_dir = "/neodata/open_dataset/mlb_data/results"
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

df_target = df[df['game_year'] == 2024].copy()

#%% batter data

df_pa = df_target.dropna(subset=['events']).copy()
df_pa = df_pa[~df_pa['events'].isin(['ejection', 'game_advisory', 'truncated_pa'])]

# 定義事件分類
hits = ['single', 'double', 'triple', 'home_run']
walks = ['walk', 'intent_walk']
strikeouts = ['strikeout', 'strikeout_double_play']
no_ab_events = walks + ['hit_by_pitch', 'sac_fly', 'sac_bunt', 'sac_bunt_double_play', 'sac_fly_double_play', 'catcher_interf']

# 標記傳統數據
df_pa['PA'] = 1
df_pa['AB'] = (~df_pa['events'].isin(no_ab_events)).astype(int)
df_pa['H'] = df_pa['events'].isin(hits).astype(int)
df_pa['2B'] = (df_pa['events'] == 'double').astype(int)
df_pa['3B'] = (df_pa['events'] == 'triple').astype(int)
df_pa['HR'] = (df_pa['events'] == 'home_run').astype(int)
df_pa['K'] = df_pa['events'].isin(strikeouts).astype(int)
df_pa['BB'] = df_pa['events'].isin(walks).astype(int)

# 標記擊球型態 (用於投手滾飛比)
df_pa['GB'] = (df_pa['bb_type'] == 'ground_ball').astype(int)
df_pa['FB'] = (df_pa['bb_type'] == 'fly_ball').astype(int)
df_pa['LD'] = (df_pa['bb_type'] == 'line_drive').astype(int)
df_pa['PU'] = (df_pa['bb_type'] == 'popup').astype(int)
df_pa['BIP'] = df_pa['GB'] + df_pa['FB'] + df_pa['LD'] + df_pa['PU']

batting_stats = df_pa.groupby(['batter']).agg(**{
    'G': ('game_pk', 'nunique'),
    'PA': ('PA', 'sum'),
    'AB': ('AB', 'sum'),
    'H': ('H', 'sum'),
    '2B': ('2B', 'sum'),
    '3B': ('3B', 'sum'),
    'HR': ('HR', 'sum'),
    'K': ('K', 'sum'),
    'BB': ('BB', 'sum')
}).reset_index()

batting_stats['AVG'] = np.where(batting_stats['AB'] > 0, batting_stats['H'] / batting_stats['AB'], 0).round(3)
batting_stats.to_csv(os.path.join(save_dir, "batting_stats.csv"), index=False)

#%%

out_1 = ['strikeout', 'field_out', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt']
out_2 = ['grounded_into_double_play', 'double_play', 'sac_fly_double_play', 'strikeout_double_play']
out_3 = ['triple_play']

df_pa['Outs_on_play'] = 0
df_pa.loc[df_pa['events'].isin(out_1), 'Outs_on_play'] = 1
df_pa.loc[df_pa['events'].isin(out_2), 'Outs_on_play'] = 2
df_pa.loc[df_pa['events'].isin(out_3), 'Outs_on_play'] = 3

if 'post_bat_score' in df_target.columns and 'bat_score' in df_target.columns:
    df_target['run_scored_on_pitch'] = df_target['post_bat_score'] - df_target['bat_score']
    runs_allowed = df_target.groupby('pitcher')['run_scored_on_pitch'].sum().reset_index()
    runs_allowed.rename(columns={'run_scored_on_pitch': 'R'}, inplace=True)
else:
    runs_allowed = pd.DataFrame(columns=['pitcher', 'R'])

pitching_results = df_pa.groupby(['pitcher']).agg(**{
    'G': ('game_pk', 'nunique'),
    'TBF': ('PA', 'sum'),    
    'AB': ('AB', 'sum'),     
    'H': ('H', 'sum'),
    'HR': ('HR', 'sum'),
    'K': ('K', 'sum'),
    'BB': ('BB', 'sum'),
    'BIP': ('BIP', 'sum'),
    'GB': ('GB', 'sum'),
    'FB': ('FB', 'sum'),
    'LD': ('LD', 'sum'),
    'Outs': ('Outs_on_play', 'sum') # 新增總出局數
}).reset_index()

pitching_results['IP'] = (pitching_results['Outs'] // 3) + (pitching_results['Outs'] % 3) / 10

# 將總失分 (R) 合併進來
if not runs_allowed.empty:
    pitching_results = pd.merge(pitching_results, runs_allowed, on='pitcher', how='left')

pitching_results['GB%'] = np.where(pitching_results['BIP'] > 0, (pitching_results['GB'] / pitching_results['BIP']) * 100, 0).round(1)
pitching_results['FB%'] = np.where(pitching_results['BIP'] > 0, (pitching_results['FB'] / pitching_results['BIP']) * 100, 0).round(1)

if 'R' in pitching_results.columns:
    pitching_results['RA9'] = np.where(pitching_results['Outs'] > 0, (pitching_results['R'] * 27) / pitching_results['Outs'], 0).round(2)

df_pitches = df_target.dropna(subset=['pitch_name']).copy()
pitch_counts = pd.crosstab(df_pitches['pitcher'], df_pitches['pitch_name']).reset_index()
pitch_counts['Total_Pitches'] = pitch_counts.drop(columns=['pitcher']).sum(axis=1)

final_pitching_stats = pd.merge(pitching_results, pitch_counts, on='pitcher', how='left')

cols_to_int = pitch_counts.columns.drop('pitcher')
final_pitching_stats[cols_to_int] = final_pitching_stats[cols_to_int].fillna(0).astype(int)

#final_pitching_stats.to_csv(os.path.join(save_dir, "pitching_stats.csv"), index=False)

#%% download data via pybaseball

from pybaseball import pitching_stats

# 抓取 2024 年的投手數據，qual=0 代表沒有局數限制
# 注意：pybaseball 預設會抓取進階 (Advanced) 數據，但裡面已經包含了 Standard 數據的大部分指標
print("Downloading FanGraphs data via pybaseball...")
df_pybaseball = pitching_stats(2024, 2024, qual=0)

print(f"成功抓取！共 {len(df_pybaseball)} 筆投手資料。")
print(df_pybaseball[['Name', 'Team', 'G', 'GS', 'IP', 'ERA']].head())


df_pybaseball.to_csv(os.path.join(save_dir, "pybaseball_pitching_stats.csv"), index=False)



#%%