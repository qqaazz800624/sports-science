#%%

import pandas as pd
import numpy as np
import os
from utils import get_expected_bases_map, Config

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
            #(df['description'] == 'hit_into_play') & 
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

valid_df = df_bip.copy()

#%%

target_year = 2021
target_team = 'MIN'

target_df = valid_df[valid_df['game_year'] == target_year].copy()

df_pa = target_df[target_df['events'].notna()].copy()

hits_events = ['single', 'double', 'triple', 'home_run']
out_events = ['strikeout', 'field_out', 'grounded_into_double_play', 'force_out', 
              'fielders_choice', 'field_error', 'strikeout_double_play', 'double_play', 
              'other_out', 'fielders_choice_out']
ab_events = hits_events + out_events

def calc_statcast_stats(data, target_team='HOU', is_target_batting=True):
    if is_target_batting:
        batting_data = data[((data['home_team'] == target_team) & (data['inning_topbot'] == 'Bot')) | 
                            ((data['away_team'] == target_team) & (data['inning_topbot'] == 'Top'))]
    else:
        batting_data = data[((data['home_team'] == target_team) & (data['inning_topbot'] == 'Top')) | 
                            ((data['away_team'] == target_team) & (data['inning_topbot'] == 'Bot'))]

    if len(batting_data) == 0:
        return {'G': 0, 'PA': 0, 'AVG': 0, 'OBP': 0, 'SLG': 0, 'OPS': 0, 'HR': 0, 'R/G': 0}

    events = batting_data['events']
    
    ab = events.isin(ab_events).sum()
    h = events.isin(hits_events).sum()
    bb = events.isin(['walk', 'intent_walk']).sum() 
    hbp = (events == 'hit_by_pitch').sum()
    sf = (events == 'sac_fly').sum()
    hr = (events == 'home_run').sum()
    
    b1 = (events == 'single').sum()
    b2 = (events == 'double').sum()
    b3 = (events == 'triple').sum()
    tb = b1 + (2 * b2) + (3 * b3) + (4 * hr)
    
    games = batting_data['game_pk'].nunique() if 'game_pk' in batting_data.columns else 0

    if 'post_bat_score' in batting_data.columns:
        total_runs = batting_data.groupby('game_pk')['post_bat_score'].max().sum()
    elif 'bat_score' in batting_data.columns:
        total_runs = batting_data.groupby('game_pk')['bat_score'].max().sum()
    else:
        total_runs = 0
        
    run_per_g = total_runs / games if games > 0 else 0

    avg = h / ab if ab > 0 else 0
    
    obp_denominator = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denominator if obp_denominator > 0 else 0
    
    slg = tb / ab if ab > 0 else 0
    ops = obp + slg

    return {
        'G': games,
        'PA': len(batting_data), 
        'AVG': round(avg, 3),
        'OBP': round(obp, 3),
        'SLG': round(slg, 3),
        'OPS': round(ops, 3),
        'HR': hr,
        'R/G': round(run_per_g, 2)  
    }


df_det_home = df_pa[df_pa['home_team'] == target_team]
df_det_away = df_pa[df_pa['away_team'] == target_team]

results = {
    f'{target_team} (Home Park)': calc_statcast_stats(df_det_home, target_team=target_team, is_target_batting=True),
    f'{target_team} (Other ballpark)': calc_statcast_stats(df_det_away, target_team=target_team, is_target_batting=True),
    f'Opponents (Home Park)': calc_statcast_stats(df_det_home, target_team=target_team, is_target_batting=False),
    f'Opponents (Other ballpark)': calc_statcast_stats(df_det_away, target_team=target_team, is_target_batting=False)
}

def calc_general_stats(batting_data):
    if len(batting_data) == 0:
        return {'G': 0, 'PA': 0, 'AVG': 0, 'OBP': 0, 'SLG': 0, 'OPS': 0, 'HR': 0, 'R/G': 0}

    events = batting_data['events']
    
    ab = events.isin(ab_events).sum()
    h = events.isin(hits_events).sum()
    bb = events.isin(['walk', 'intent_walk']).sum() 
    hbp = (events == 'hit_by_pitch').sum()
    sf = (events == 'sac_fly').sum()
    hr = (events == 'home_run').sum()
    
    b1 = (events == 'single').sum()
    b2 = (events == 'double').sum()
    b3 = (events == 'triple').sum()
    tb = b1 + (2 * b2) + (3 * b3) + (4 * hr)
    
    games = batting_data['game_pk'].nunique() if 'game_pk' in batting_data.columns else 0


    if 'post_bat_score' in batting_data.columns:
        total_runs = batting_data.groupby(['game_pk', 'inning_topbot'])['post_bat_score'].max().sum()
    elif 'bat_score' in batting_data.columns:
        total_runs = batting_data.groupby(['game_pk', 'inning_topbot'])['bat_score'].max().sum()
    else:
        total_runs = 0
        
    
    team_games = batting_data.groupby('game_pk')['inning_topbot'].nunique().sum()
    run_per_g = total_runs / team_games if team_games > 0 else 0
    
    avg = h / ab if ab > 0 else 0
    obp_denominator = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denominator if obp_denominator > 0 else 0
    slg = tb / ab if ab > 0 else 0
    ops = obp + slg

    return {
        'G': games, 
        'PA': len(batting_data), 
        'AVG': round(avg, 3),
        'OBP': round(obp, 3),
        'SLG': round(slg, 3),
        'OPS': round(ops, 3),
        'HR': hr,
        'R/G': round(run_per_g, 2)  
    }

df_target_home = df_pa[df_pa['home_team'] == target_team]

results[f'All teams at {target_team} Home Park'] = calc_general_stats(df_target_home)
results['League Average'] = calc_general_stats(df_pa)

final_summary_df = pd.DataFrame.from_dict(results, orient='index')
print(final_summary_df)

#%%

save_dir = "/neodata/open_dataset/mlb_data/results"
final_summary_df.to_csv(f'{save_dir}/{target_team}_statcast_summary_{target_year}.csv')





#%%







#%%











#%%







#%%








#%%







#%%