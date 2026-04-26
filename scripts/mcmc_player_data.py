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

data_dir = "/Users/wujhejia/Documents/sports-science/data/preprocessed"
save_dir = "/Users/wujhejia/Documents/sports-science/data/results"
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

#%%

df_terminal = df_target.dropna(subset=['events']).copy()
df_terminal = df_terminal[df_terminal['events'] != 'truncated_pa']

event_mapping = {
    'single': 'single',
    'double': 'double',
    'triple': 'triple',
    'home_run': 'home_run',
    
    'walk': 'walk',
    'intent_walk': 'walk',         
    'hit_by_pitch': 'walk',        
    'catcher_interf': 'walk',      
    
    'strikeout': 'out',
    'field_out': 'out',
    'force_out': 'out',
    'grounded_into_double_play': 'out',
    'double_play': 'out',
    'sac_fly': 'out',
    'sac_bunt': 'out',
    'fielders_choice': 'out',
    'fielders_choice_out': 'out',
    'field_error': 'out',          
    'sac_fly_double_play': 'out',
    'strikeout_double_play': 'out',
    'triple_play': 'out'
}

df_terminal['simple_event'] = df_terminal['events'].map(event_mapping).fillna('out')

player_probs_df = (
    df_terminal.groupby('batter')['simple_event']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

required_columns = ['out', 'walk', 'single', 'double', 'triple', 'home_run']
for col in required_columns:
    if col not in player_probs_df.columns:
        player_probs_df[col] = 0.0

player_probs_df = player_probs_df[required_columns]

#%%


print(player_probs_df.head(20))




#%%