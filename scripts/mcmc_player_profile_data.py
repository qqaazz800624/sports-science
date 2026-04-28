#%%

import pandas as pd
import numpy as np
import os
from utils import get_expected_bases_map, Config
from tqdm import tqdm
from pybaseball import playerid_reverse_lookup
import json

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

def add_batter_names_to_statcast(df):
    
    print("start loading batter names...")
    
    unique_batter_ids = df['batter'].dropna().unique().tolist()
    
    id_map = playerid_reverse_lookup(unique_batter_ids, key_type='mlbam')
    
    id_map['batter_name'] = (id_map['name_first'].str.title() + ' ' + 
                             id_map['name_last'].str.title())
    
    id_map_clean = id_map[['key_mlbam', 'batter_name']]
    
    df_merged = df.merge(id_map_clean, left_on='batter', right_on='key_mlbam', how='left')
    
    df_merged = df_merged.drop(columns=['key_mlbam'])
    
    print(f"successfully loaded names for {len(unique_batter_ids)} batters!")
    return df_merged


exp_map = get_expected_bases_map(config=config)
df = pd.read_parquet(truncated_file_path)

df = add_batter_names_to_statcast(df)
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
    df_terminal.groupby('batter_name')['simple_event']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

required_columns = ['out', 'walk', 'single', 'double', 'triple', 'home_run']
for col in required_columns:
    if col not in player_probs_df.columns:
        player_probs_df[col] = 0.0

player_probs_df = player_probs_df[required_columns]



target_players = {
    'Shohei Ohtani': 'Ohtani',
    'Mookie Betts': 'Betts',
    'Freddie Freeman': 'Freeman',
    'Teoscar Hernández': 'Teoscar',
    'Max Muncy': 'Muncy',
    'Will Smith': 'Smith',
    'Gavin Lux': 'Lux',
    'Tommy Edman': 'Edman',
    'Miguel Rojas': 'Rojas'
}

player_profiles_updated = []

print("Start updating player profiles with 2024 data...")

for i, (full_name, short_name) in enumerate(target_players.items()):
    if full_name in player_probs_df.index:
        row = player_probs_df.loc[full_name]
        
        bb_prob = round(row['walk'], 3).item()
        b1_prob = round(row['single'], 3).item()
        b2_prob = round(row['double'], 3).item()
        b3_prob = round(row['triple'], 3).item()
        hr_prob = round(row['home_run'], 3).item()
        
        out_prob = round(1.0 - (bb_prob + b1_prob + b2_prob + b3_prob + hr_prob), 3)
        
        profile = {
            'id': i,
            'name': short_name,
            'OUT': out_prob,
            'BB': bb_prob,
            '1B': b1_prob,
            '2B': b2_prob,
            '3B': b3_prob,
            'HR': hr_prob
        }
        player_profiles_updated.append(profile)
    else:
        print(f"Cannot find player: {full_name}")

print("\nplayer_profiles = [")
for p in player_profiles_updated:
    print(f"    {p},")
print("]")

with open(os.path.join(save_dir, "player_profiles.json"), 'w') as f:
    json.dump(player_profiles_updated, f, indent=4)

#%%

print(player_probs_df.head(20))




#%%