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

BASE_DIR_tm = "/Users/yantianli/factor-and-defense-factor/"

def generate_summary_tbl(data:pd.DataFrame,
                        team: str,
                        year: int)-> dict:
        """
        給定該隊該年整理好的 event table（包含 sum_real_count 與 sum_expected_count），
        回傳該隊年度的成績（expected & real）。
        """
        # --- 預期值 (Expected) ---
        ex_pa, ex_ab = cal_pa_ab(data, col='sum_expected_count')
        ex_ops = cal_ops(data, col='sum_expected_count')
        ex_slg = cal_slg(data, col='sum_expected_count')
        ex_obp = cal_obp(data, col='sum_expected_count')
        ex_babip = cal_babip(data, col='sum_expected_count')

        # --- 實際值 (Real) ---
        real_pa, real_ab = cal_pa_ab(data, col='sum_real_count')
        real_ops = cal_ops(data, col='sum_real_count')
        real_slg = cal_slg(data, col='sum_real_count')
        real_obp = cal_obp(data, col='sum_real_count')
        real_babip = cal_babip(data, col='sum_real_count')

        # --- 整合成一筆 DataFrame ---
        return {
        "Team": team,
        "Year": year,
        # 預期值
        "ex_PA": ex_pa, "ex_AB": ex_ab,
        "ex_OPS": ex_ops, "ex_SLG": ex_slg, "ex_OBP": ex_obp, "ex_BABIP": ex_babip,
        # 實際值
        "real_PA": real_pa, "real_AB": real_ab,
        "real_OPS": real_ops, "real_SLG": real_slg, "real_OBP": real_obp, "real_BABIP": real_babip
        }

def get_team_score(score_type: str):
    """
    score_type 可為 'bat', 'pitch', 'park'
    """
    path = os.path.join(BASE_DIR_tm, f"team_{score_type}_tbl.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {score_type} 資料：{path}")
    print(f"📦 載入 {score_type} 資料：{path}")

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    # test1 = pd.read_parquet("/Users/yantianli/Desktop/mlbdata/team_pitching_tbl/NYM_2015_pitcher.parquet")
    # test2 = pd.read_parquet("/Users/yantianli/Desktop/mlbdata/team_batting_tbl/NYM_2015_batter.parquet")

    # dp(test1.head())
    # dp(test2.head())

    df = get_truncated_dataset_with_team()
    rtheta_distri = get_rtheta_prob_tbl()
    df = df[df['game_type'] == 'R'] # 限制在 regular season


    team_list = df['home_team'].unique().tolist()
    year = df['game_year'].unique().tolist()

    # 可生成三種類型 summary：batting、pitching、park
    batting_summary = []
    pitching_summary = []
    park_summary = []

    for summary_type in ["batting", "pitching", "park"]:
        print(f"\n=== 開始處理 {summary_type} summary ===")
        for team in team_list:
            for yr in [y for y in year if y not in [2014, 2020]]:
                if summary_type == "batting": # 隊伍打者資料
                    file_path = f"/Users/yantianli/Desktop/mlbdata/team_batting_tbl/{team}_{yr}_batter.parquet"
                elif summary_type == "pitching": # 隊伍投手資料
                    file_path = f"/Users/yantianli/Desktop/mlbdata/team_pitching_tbl/{team}_{yr}_pitcher.parquet"
                elif summary_type == "park": # 球隊主場資料
                    # /Users/yantianli/Desktop/mlbdata/park_tbl/ATL_2015_home.parquet
                    file_path = f"/Users/yantianli/Desktop/mlbdata/park_tbl/{team}_{yr}_home.parquet"
                else:
                    continue

                if not os.path.exists(file_path):
                    print(f"⚠️ 找不到 {summary_type} 檔案：{file_path}")
                    continue

                data_tbl = pd.read_parquet(file_path)
                print(f"正在處理 {summary_type}：{team} {yr}")
                summary = generate_summary_tbl(data_tbl, team, yr)
                if summary_type == "batting":
                    batting_summary.append(summary)
                elif summary_type == "pitching":
                    pitching_summary.append(summary)
                elif summary_type == "park":
                    park_summary.append(summary)

    # 輸出 CSV 檔案
    if batting_summary:
        batting_df = pd.DataFrame(batting_summary)
        bat_save_path = "/Users/yantianli/factor-and-defense-factor/team_bat_tbl.csv"
        batting_df.to_csv(bat_save_path, index=False)
        print(f"成功儲存球隊打擊成績：{bat_save_path}")
    if pitching_summary:
        pitching_df = pd.DataFrame(pitching_summary)
        pitch_save_path = "/Users/yantianli/factor-and-defense-factor/team_pitch_tbl.csv"
        pitching_df.to_csv(pitch_save_path, index=False)
        print(f"成功儲存球隊投手成績：{pitch_save_path}")
    if park_summary:
        park_df = pd.DataFrame(park_summary)
        park_save_path = "/Users/yantianli/factor-and-defense-factor/team_park_tbl.csv"
        park_df.to_csv(park_save_path, index=False)
        print(f"成功儲存球場成績：{park_save_path}")

#%%
# def get_tm_park_score():
#     """回傳 15~24的球場成績"""
#     path = os.path.join(BASE_DIR_tm, "team_park_tbl.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"找不到 truncated 資料：{path}")
#     print(f"📦 載入 truncated 資料：{path}")

#     df = pd.read_csv(path, index_col=0)
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     df = df.reset_index(drop=True)
#     return df


# def get_tm_batting_score():
#     """回傳 15~24的球隊打擊成績"""
#     path = os.path.join(BASE_DIR_tm, "team_bat_tbl.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"找不到 truncated 資料：{path}")
#     print(f"📦 載入 truncated 資料：{path}")

#     df = pd.read_csv(path, index_col=0)
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     df = df.reset_index(drop=True)
#     return df

# def get_tm_pitching_score():
#     """回傳 15~24的球隊投手成績"""
#     path = os.path.join(BASE_DIR_tm, "team_pitch_tbl.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"找不到 truncated 資料：{path}")
#     print(f"📦 載入 truncated 資料：{path}")

#     df = pd.read_csv(path, index_col=0)
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     df = df.reset_index(drop=True)
#     return df
#%%