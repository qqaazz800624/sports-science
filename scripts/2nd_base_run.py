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
#df_target = df[df['game_year'] == 2024].copy()
df_target = df.copy()

#%%


def calculate_baserunning_prob(data):
    print("正在計算進階跑壘機率...")
    # 確保分數欄位沒有缺失值
    df_calc = data.copy()
    df_calc['bat_score'] = df_calc['bat_score'].fillna(0)
    df_calc['post_bat_score'] = df_calc['post_bat_score'].fillna(0)
    
    # 1. 篩選出所有「一壘安打」的打席
    df_single = df_calc[df_calc['events'] == 'single'].copy()
    
    # 2. 情境 A：二壘有人，且三壘「無人」
    sit_A = df_single[df_single['on_2b'].notna() & df_single['on_3b'].isna()].copy()
    sit_A['runs_scored'] = sit_A['post_bat_score'] - sit_A['bat_score']
    sit_A['runner_scored'] = sit_A['runs_scored'] >= 1
    
    # 3. 情境 B：二壘有人，且三壘「有人」
    sit_B = df_single[df_single['on_2b'].notna() & df_single['on_3b'].notna()].copy()
    sit_B['runs_scored'] = sit_B['post_bat_score'] - sit_B['bat_score']
    # 因為三壘跑者會拿 1 分，所以得分必須 >= 2 才能證明二壘跑者也回來了
    sit_B['runner_scored'] = sit_B['runs_scored'] >= 2
    
    # 4. 統計與計算機率
    total_opportunities = len(sit_A) + len(sit_B)
    total_success = sit_A['runner_scored'].sum() + sit_B['runner_scored'].sum()
    
    prob = total_success / total_opportunities if total_opportunities > 0 else 0
    
    print("-" * 40)
    print("二壘跑者在一壘安打時的得分機率")
    print("-" * 40)
    print(f"總共發生次數: {total_opportunities} 次")
    print(f"成功衝回本壘: {total_success} 次")
    print(f"停在三壘或出局: {total_opportunities - total_success} 次")
    print(f"實戰得分機率: {prob:.2%}")
    print("-" * 40)
    
    return prob

# 將 2024 年的資料餵進去算
p_1B_score_from_2nd = calculate_baserunning_prob(df_target)




#%%








#%%











#%%







#%%








#%%