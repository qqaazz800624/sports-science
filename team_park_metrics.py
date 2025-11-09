#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

# from pybaseball import playerid_reverse_lookup, batting_stats, pitching_stats
# from calculate_score import combined_score_tbl, batter_data_fg, pitcher_data_fg
from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl
from calculate_offensive_mertics import cal_pa_ab, cal_babip, cal_obp, cal_ops, cal_pa_ab, cal_slg

df = get_truncated_dataset_with_team()
rtheta_distri = get_rtheta_prob_tbl()
df = df[df['game_type'] == 'R'] # 限制在 regular season


team_list = df['home_team'].unique().tolist()
year = df['game_year'].unique().tolist()

park_summary = []

for team in team_list:
    for yr in [y for y in year if y != 2014]:
        file_path = f'/Users/yantianli/Desktop/mlbdata/park_tbl/{team}_{yr}_home.parquet'
        park_tbl = pd.read_parquet(file_path)
        
        # --- 預期值 (Expected) ---
        ex_pa, ex_ab = cal_pa_ab(park_tbl, col='sum_expected_count')
        ex_ops = cal_ops(park_tbl, col='sum_expected_count')
        ex_slg = cal_slg(park_tbl, col='sum_expected_count')
        ex_obp = cal_obp(park_tbl, col='sum_expected_count')
        ex_babip = cal_babip(park_tbl, col='sum_expected_count')

        # --- 實際值 (Real) ---
        real_pa, real_ab = cal_pa_ab(park_tbl, col='sum_real_count')
        real_ops = cal_ops(park_tbl, col='sum_real_count')
        real_slg = cal_slg(park_tbl, col='sum_real_count')
        real_obp = cal_obp(park_tbl, col='sum_real_count')
        real_babip = cal_babip(park_tbl, col='sum_real_count')

        # --- 整合成一筆 DataFrame ---
        team_data = {
            "Team": team,
            "Year": yr,
            # 預期值
            "ex_PA": ex_pa,
            "ex_AB": ex_ab,
            "ex_OPS": ex_ops,
            "ex_SLG": ex_slg,
            "ex_OBP": ex_obp,
            "ex_BABIP": ex_babip,
            # 實際值
            "real_PA": real_pa,
            "real_AB": real_ab,
            "real_OPS": real_ops,
            "real_SLG": real_slg,
            "real_OBP": real_obp,
            "real_BABIP": real_babip
        }

        park_summary.append(team_data)

# tm_bat_df = pd.DataFrame(park_summary)
# tm_bat_df.to_csv('/Users/yantianli/factor_and_defense_factor/tm_park_tbl.csv')


BASE_DIR_tm = "/Users/yantianli/factor_and_defense_factor/"


def get_tm_park_score():
    """回傳 15~24的球場打擊成績"""
    path = os.path.join(BASE_DIR_tm, "tm_park_tbl.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 truncated 資料：{path}")
    print(f"📦 載入 truncated 資料：{path}")

    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print("✅ 自動移除 Unnamed 欄完成")
    df = df.reset_index(drop=True)
    return df


def get_tm_batting_score():
    """回傳 15~24的球隊打擊成績"""
    path = os.path.join(BASE_DIR_tm, "tm_bat_tbl.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 truncated 資料：{path}")
    print(f"📦 載入 truncated 資料：{path}")

    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print("✅ 自動移除 Unnamed 欄完成")
    df = df.reset_index(drop=True)
    return df

def get_tm_pitching_score():
    """回傳 15~24的球隊投手成績"""
    path = os.path.join(BASE_DIR_tm, "tm_pitch_tbl.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 truncated 資料：{path}")
    print(f"📦 載入 truncated 資料：{path}")

    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print("✅ 自動移除 Unnamed 欄完成")
    df = df.reset_index(drop=True)
    return df



#%%