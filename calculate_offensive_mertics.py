#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

from pybaseball import playerid_reverse_lookup

from expect_score import get_expected_dataset, get_event_distribution, get_whole_dataset
from calculate_score import hip_score_tbl, nonhip_score_tbl, ibb_score_tbl, combined_score_tbl

print(get_expected_dataset())
#%%
def cal_pa_ab(df: pd.DataFrame):
    """
    計算打席數 (PA) 與打數 (AB)。
    
    PA：所有 sum_real_count 的總和。
    AB：PA 扣除不計入打數的事件（如 'walk', 'sac_bunt', 'sac_fly', 'catcher_interf', 'hit_by_pitch'）。
    
    Parameters:
    df (pd.DataFrame): 來自 combined_score_tbl 的結果，需包含 'events' 與 'sum_real_count'
    Returns:
    tuple: (pa, ab)，分別為打席數和打數。
    """

    # 定義不計入打數 (AB) 的事件
    exclude_events = ['IBB','walk', 'sac_bunt', 'sac_fly', 'catcher_interf', 'hit_by_pitch', 'sac_bunt_double_play', 'sac_fly_double_play']

    # 確保 sum_real_count 是 int，並填補 NaN 為 0
    df['sum_real_count'] = pd.to_numeric(df['sum_real_count'], errors='coerce').fillna(0)

    all_events = [
        'strikeout', 'strikeout_double_play', 'walk', 'intent_walk', 'hit_by_pitch',
        'IBB', 'single', 'double', 'triple', 'home_run',
        'field_out', 'double_play', 'triple_play',
        'sac_fly', 'sac_bunt', 'catcher_interf',
        'sac_bunt_double_play', 'sac_fly_double_play'
    ]

    # 若某些事件不存在於 DataFrame，補上 0
    for evt in all_events:
        if evt not in df['events'].values:
            df.loc[len(df)] = [evt, 0, 0] if 'sum_expected_count' in df.columns else [evt, 0]

    # 不計入打數的事件
    exclude_events = [
        'IBB', 'walk', 'intent_walk', 'sac_bunt', 'sac_fly',
        'catcher_interf', 'hit_by_pitch', 'sac_bunt_double_play', 'sac_fly_double_play'
    ]

    # 計算 PA（打席數）= 總事件 - intentional walk
    intent_walk = df.loc[df['events'] == 'intent_walk', 'sum_real_count'].sum()
    pa = df['sum_real_count'].sum() - intent_walk

    # 計算不算入打數的事件
    excluded_count = df.loc[df['events'].isin(exclude_events), 'sum_real_count'].sum()

    # 計算 AB
    ab = pa - excluded_count

    return pa, ab



judge_df = combined_score_tbl(
    data=get_expected_dataset(),
    dist_df=get_event_distribution(),
    year=2019,
    player_mlbid=592450,
    player_type='batter'
)

pa, ab = cal_pa_ab(judge_df)
print(f"Judge 2019 年 PA: {pa}, AB: {ab}")
#%%