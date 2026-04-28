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

exp_map = get_expected_bases_map(config=config)
df = pd.read_parquet(truncated_file_path)

#df_target = df[df['game_year'] == 2024].copy()
df_target = df.copy()


#%%

def calculate_1b_score_on_double_prob(data):
    print("正在計算進階跑壘機率 (一壘跑者遇二壘安打)...")
    df_calc = data.copy()
    df_calc['bat_score'] = df_calc['bat_score'].fillna(0)
    df_calc['post_bat_score'] = df_calc['post_bat_score'].fillna(0)
    
    # 1. 篩選出所有「二壘安打」的打席
    df_double = df_calc[df_calc['events'] == 'double'].copy()
    
    # 2. 篩選出「一壘有人」的情境
    df_double = df_double[df_double['on_1b'].notna()].copy()
    
    # 計算該打席的總得分
    df_double['runs_scored'] = df_double['post_bat_score'] - df_double['bat_score']
    
    # 3. 計算在該一壘跑者「前面」有幾個跑者 (二壘或三壘有人)
    # True 會被轉為 1，False 會被轉為 0
    runners_ahead = df_double['on_2b'].notna().astype(int) + df_double['on_3b'].notna().astype(int)
    
    # 4. 如果得分大於「前面的跑者數量」，代表一壘跑者也衝回來得分了
    df_double['runner_scored'] = df_double['runs_scored'] >= (runners_ahead + 1)
    
    # 5. 統計與計算機率
    total_opportunities = len(df_double)
    total_success = df_double['runner_scored'].sum()
    
    prob = total_success / total_opportunities if total_opportunities > 0 else 0
    
    print("-" * 40)
    print("一壘跑者在二壘安打時的得分機率")
    print("-" * 40)
    print(f"總共發生次數: {total_opportunities} 次")
    print(f"成功衝回本壘: {total_success} 次")
    print(f"停在三壘或出局: {total_opportunities - total_success} 次")
    print(f"實戰得分機率: {prob:.2%}")
    print("-" * 40)
    
    return prob

# 將資料餵進去算
p_2B_score_from_1st = calculate_1b_score_on_double_prob(df_target)




#%%







#%%






#%%







#%%






#%%