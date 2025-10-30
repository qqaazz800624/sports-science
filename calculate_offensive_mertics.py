#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

from expect_score import get_truncated_dataset, get_rtheta_prob_tbl, get_whole_dataset
from calculate_score import combined_score_tbl


def cal_pa_ab(df: pd.DataFrame,
              col: str = 'sum_real_count'):
    """
    計算打席數 (PA) 與打數 (AB)。
    
    PA：所有 sum_real_count 的總和。
    AB：PA 扣除不計入打數的事件（如 'walk', 'sac_bunt', 'sac_fly', 'catcher_interf', 'hit_by_pitch'）。
    
    Parameters:
    df (pd.DataFrame): 來自 combined_score_tbl 的結果，需包含 'events' 與 'sum_real_count'
    column (str): 指定計算實際結果 ('sum_real_count') 或 預期結果 ('sum_expected_count')。

    Returns:
    tuple: (pa, ab)，分別為打席數和打數。
    """

    # 定義不計入打數 (AB) 的事件
    exclude_events = ['IBB','walk', 'sac_bunt', 'sac_fly', 'catcher_interf', 'hit_by_pitch', 'sac_bunt_double_play', 'sac_fly_double_play']

    # 確保 sum_real_count 是 int，並填補 NaN 為 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    all_events = [
        'strikeout', 'strikeout_double_play', 'walk', 'intent_walk', 'hit_by_pitch',
        'single', 'double', 'triple', 'home_run',
        'field_out', 'double_play', 'triple_play',
        'sac_fly', 'sac_bunt', 'catcher_interf',
        'sac_bunt_double_play', 'sac_fly_double_play'
    ]

    # 若某些事件不存在於 DataFrame，補上 0
    for evt in all_events:
        if evt not in df['events'].values:
            df.loc[len(df)] = [evt, 0, 0] if col in df.columns else [evt, 0]

    # 不計入打數的事件
    exclude_events = ["intent_walk",
        'walk', 'sac_bunt', 'sac_fly',
        'catcher_interf', 'hit_by_pitch', 'sac_bunt_double_play', 'sac_fly_double_play'
    ]

    # 計算 PA（打席數）
    pa = df[col].sum()

    # 計算不算入打數的事件
    excluded_count = df.loc[df['events'].isin(exclude_events), col].sum()

    # 計算 AB
    ab = pa - excluded_count

    return round(pa, 4), round(ab, 4)

def cal_ba(df: pd.DataFrame,
           col: str = 'sum_real_count'):
    """
    Parameters:
    combined_df (pd.DataFrame): 包含 'events' 和 'sum_real_count' 或 'sum_expected_count' 的 DataFrame。
    column (str): 指定計算實際結果 ('sum_real_count') 或 預期結果 ('sum_expected_count')。

    Returns:
    float: 打擊率 (四捨五入到小數點第四位)。
    """

    # Calculte AB
    _, ab = cal_pa_ab(df, col)
   
    # 確保數據為數值類型，並填補 NaN 為 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 計算安打數 (H)
    hit_events = ['single', 'double', 'triple', 'home_run']
    H = df.loc[df['events'].isin(hit_events), col].sum()

    if ab == 0:
        return 0.0
    
    ba = H / ab
    return round(ba, 4)

def cal_obp(df: pd.DataFrame,
            col: str = 'sum_real_count'):
    """
    計算上壘率 (OBP, On-Base Percentage)
    OBP = (H + BB + HBP) / (AB + BB + HBP + Sac flies)

    Parameters:
    df (pd.DataFrame): 包含 'events' 和 'sum_real_count' 或 'sum_expected_count' 的 DataFrame。
    column (str): 指定計算實際結果 ('sum_real_count') 或 預期結果 ('sum_expected_count')。

    Returns:
    float: 上壘率 (四捨五入到小數點第四位)。
    """
    # Calculte pa
    _, ab = cal_pa_ab(df, col)

    # 確保數據為數值類型，並填補 NaN 為 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 計算安打數 (H)
    hit_events = ['single', 'double', 'triple', 'home_run']
    H = df.loc[df['events'].isin(hit_events), col].sum()

    #計算 BB & HBP & IBB & sac flies
    BB = df.loc[df['events'] == 'walk', col].sum()
    HBP = df.loc[df['events'] == 'hit_by_pitch', col].sum()
    IBB = df.loc[df['events'] == 'intent_walk', col].sum()
    SF = df.loc[df['events'] == 'sac_fly', col].sum()
    
    # 避免除以零
    denom = ab + BB + HBP + SF + IBB
    if denom == 0:
        return 0.0

    # 計算上壘率 OBP 
    obp = (H + BB + HBP + IBB) / denom
    return round(obp, 4)
    
def cal_slg(df: pd.DataFrame, 
                  col: str = 'sum_real_count'):
    """
    計算長打率 (SLG)：
      SLG = Total Bases / AB
    """
    _, ab = cal_pa_ab(df, col)
    if ab == 0:
        return 0.0
    
    weights = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    
    total_bases = sum(df.loc[df['events'] == hit, col].sum() * weight
                      for hit, weight in weights.items())

    # 計算 SLG = 總壘打數 / AB
    slg = total_bases / ab

    return round(slg, 4)

def cal_babip(df: pd.DataFrame, 
              col: str):
    """
    計算打擊率 BAbip
    
    Parameters:
    df (pd.DataFrame): 包含 'events' 和 'sum_real_count' 或 'sum_expected_count' 的 DataFrame。
    col (str): 指定計算實際結果 ('sum_real_count') 或 預期結果 ('sum_expected_count')。

    Returns:
    float: 打擊率 (四捨五入到小數點第四位)。
    """

    # 先計算 AB (從 calculate_pa_ab)
    _, ab = cal_pa_ab(df, col)

    # 確保數據為數值類型，並填補 NaN 為 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 計算安打數 (H)
    hit_events = ['single', 'double', 'triple', 'home_run']
    H = df.loc[df['events'].isin(hit_events), col].sum()
    # calculate HR
    HR = df.loc[df['events'] == 'home_run', col].sum()
    # calculate stike out
    SO = df.loc[df['events'] == 'strikeout', col].sum()
    # calculate sarc fly
    SF = df.loc[df['events'] == 'sac_fly', col].sum()
    # 避免除以零
    if ab == 0:
        return 0.0

    # 計算 babip
    babip = (H - HR)/ (ab - SO - HR + SF)
    
    return round(babip, 4)

def cal_ops(df: pd.DataFrame,
            col: str = 'sum_real_count'):
    """
    計算整體進攻指標 OPS 

    OPS = OBP + SLG

    Parameters:
    df (pd.DataFrame): 含 'events' 和統計欄位的 DataFrame
    col (str): 指定計算使用 'sum_real_count' 或 'sum_expected_count'

    Returns:
    float: OPS (四捨五入到小數點第四位)。
    """
    obp = cal_obp(df, col)
    slg = cal_slg(df, col)
    ops = obp + slg

    return round(ops, 4)



year_search = 2024

judge_df = combined_score_tbl(
    data=get_truncated_dataset(),
    dist_df=get_rtheta_prob_tbl(),
    year=year_search,
    player_mlbid=592450,
    player_type='batter'
)

babip = cal_babip(judge_df, 'sum_real_count')
display(f"Judge {year_search} 年的 OBP is {babip}")
# display(judge_df)
# pa, ab = cal_pa_ab(judge_df, 'sum_expected_count')
# print(f"Judge {year_search} 年 PA: {pa}, AB: {ab}")
# ba = cal_ba(judge_df, 'sum_real_count')
# display(f"Judge {year_search} 年的 Batting Average is {ba}")
#%%