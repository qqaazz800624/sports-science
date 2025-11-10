#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl, get_whole_dataset
from team_park_metrics import get_team_score
from league_score_tbl import get_league_tbl


df = get_truncated_dataset_with_team().copy()
dist_df = get_rtheta_prob_tbl()

bat_df = get_team_score("bat")
pitch_df = get_team_score("pitch")
park_df = get_team_score("park")
league_summary_tbl = get_league_tbl()


batter_tm_col = df.pop('batter_team')
pitcher_tm_col = df.pop('pitcher_team')

new_batter_tm_col = df.columns.get_loc('batter') + 1 #type: ignore
new_pitcher_tm_col = df.columns.get_loc('pitcher') + 1 #type: ignore

df.insert(new_batter_tm_col, 'batter_team', batter_tm_col) #type: ignore
df.insert(new_pitcher_tm_col, 'pitcher_team', pitcher_tm_col) #type: ignore


# 計算打線在其他park作客的年度結果
def generate_away_park_defense_eqn(
        data:pd.DataFrame,
        batter_data:pd.DataFrame,
        league_tbl:pd.DataFrame,
        metric:str,
        yr:int):
    
    # 只保留例行賽的 data
    df = data[data['game_type'] == 'R'].copy()
    tm_list = df['away_team'].unique()
    # ---- 計算聯盟整體校正值 ----
    league_correction = (
        league_tbl.loc[league_tbl['Year']==yr, f'ex_{metric}'].values[0] 
                        - league_tbl.loc[league_tbl['Year']==yr, f'real_{metric}'].values[0]
    )
    away_eqns = {}
    for tm in tm_list:

        # 打線在客場作戰的hit into play的年度 data frame
        away_hip_df = data[
            (data['game_year'] == yr)&
            (data['away_team'] == tm)&
            (data['batter_team'] == tm)&
            (data['description'] == 'hit_into_play')] 
        if len(away_hip_df) == 0:
            continue
        # 各主場的打進場次數
        ratio_away_play = (
            away_hip_df.groupby('home_team')['description']
            .count()
            .reset_index(name='hip_count')
        )

        # 計算比例並正規化
        ratio_away_play['ratio'] = ratio_away_play['hip_count'] / ratio_away_play['hip_count'].sum()
        ratio_away_play['ratio'] = ratio_away_play['ratio'].round(4)
        ratio_away_play['ratio'] /= ratio_away_play['ratio'].sum()

        # ---- 計算最終 y 值 ----
        # 計算球隊校正值
        batter_row = batter_data.loc[
            (batter_data["Team"] == tm) & (batter_data["Year"] == yr)
        ]
        park_correction = batter_row[f'ex_{metric}'].values[0] - batter_row[f'real_{metric}'].values[0]

        y_value = park_correction - league_correction

        # ----- 建立方程式： y = yearfactor + base_term + summation coeff * defense_team -----
        base_term = f"park_factor_{tm}_{yr}"
        terms = [f"{row['ratio']:.4f} * defense_{row['home_team']}_{yr}" 
                for _, row in ratio_away_play.iterrows()]
        year_factor = "year_factor"
        # 組合成完整方程
        eq = f"{y_value:.4f} = {year_factor} + {base_term} + " + " + ".join(terms)
        away_eqns[tm] = eq
    return away_eqns



def generate_park_eqn(
            data:pd.DataFrame,
            park_data:pd.DataFrame,
            league_tbl:pd.DataFrame,
            metric:str,
            yr:int):
    # 只保留例行賽的 data
    df = data[data['game_type'] == 'R'].copy()
    tm_list = df['home_team']

    park_eqns = {}
    
    # ---- 計算聯盟整體校正值 ----
    league_correction = (
        league_tbl.loc[league_tbl['Year']==yr, f'ex_{metric}'].values[0] 
                        - league_tbl.loc[league_tbl['Year']==yr, f'real_{metric}'].values[0]
    )

    # ---- 對每個球場建立 equation ----
    for tm in tm_list.unique():
        # 篩出主場資料
        home_team_df = df[
            (df['game_year'] == yr) &
            (df['home_team'] == tm) &
            (df['description'] == 'hit_into_play')
        ]

        # 計算總打進場
        total_play = len(home_team_df)
        if total_play == 0:
            continue

        #----- 計算各隊在該主場的打進場比例 -----
        away_play_count = (
            home_team_df.groupby('batter_team')['description']
            .count()
            .reset_index(name='play_count')
        )

        # 轉成比例
        away_play_count['ratio'] = away_play_count['play_count'] / away_play_count['play_count'].sum()

        # Round & Normalize
        away_play_count['ratio'] = away_play_count['ratio'].round(4)
        away_play_count['ratio'] /= away_play_count['ratio'].sum()

        # ---- 計算球隊校正值 ----
        park_row = park_data.loc[
            (park_data["Team"] == tm) & (park_data["Year"] == yr)
        ]
        park_correction = park_row[f'ex_{metric}'].values[0] - park_row[f'real_{metric}'].values[0]

        # ---- 計算最終 y 值 ----
        y_value = park_correction - league_correction

        # 建立方程式： y = base_term + summation coeff * defense_team
        base_term = f"park_factor_{tm}_{yr}"
        terms = [f"{row['ratio']:.4f} * defense_{row['batter_team']}_{yr}" 
                for _, row in away_play_count.iterrows()]
        year_factor = "year_factor"
        # 組合成完整方程
        eq = f"{y_value:.4f} = {year_factor} + {base_term} + " + " + ".join(terms)
        park_eqns[tm] = eq

    return park_eqns

def generate_home_park_defense_eqn(
            pitch_data:pd.DataFrame,
            league_tbl:pd.DataFrame,
            metric:str,
            yr:int):
    
    # ---- 計算聯盟整體校正值 ----
    league_correction = (
        league_tbl.loc[league_tbl['Year']==yr, f'ex_{metric}'].values[0] 
                        - league_tbl.loc[league_tbl['Year']==yr, f'real_{metric}'].values[0]
    )
    team_pk_def_eqns = {}

    # ---- 計算球隊校正值 ----
    for tm in pitch_data['Team'].unique():

        pitch_row = pitch_data.loc[
            (pitch_data["Team"] == tm) & (pitch_data["Year"] == yr)
        ]
        park_correction = pitch_row[f'ex_{metric}'].values[0] - pitch_row[f'real_{metric}'].values[0]

        # ---- 計算最終 y 值 ----
        y_value = park_correction - league_correction

        # -----建立方程式： y = year_factor + park_factor + defense_factor -----
        pk_factor_term = f"park_factor_{tm}_{yr}"
        tm_defense_term = f"defense_{tm}_{yr}"
        year_factor = "year_factor"
        # 組合方程式
        eq = f"{y_value:.4f} = {year_factor} + {pk_factor_term} + {tm_defense_term}"
        team_pk_def_eqns[tm] = eq
    return team_pk_def_eqns

def collect_eqns(
        data: pd.DataFrame,
        park_data: pd.DataFrame,
        pitch_data: pd.DataFrame,
        batter_data: pd.DataFrame,
        league_tbl: pd.DataFrame,
        metric: str,
        yr:int):
    
    """
    統一生成某年度的三種類型方程式：
    - 球場因子方程式（park factor）
    - 主場防守方程式（home defense）
    - 客場打線方程式（away offense）
    """
    year_park_eqs = generate_park_eqn(
        data=data,
        park_data=park_data,
        league_tbl=league_tbl,
        metric=metric,
        yr=yr)
    
    year_home_defense_eqs = generate_home_park_defense_eqn(
        pitch_data=pitch_data,
        league_tbl=league_tbl,
        metric=metric,
        yr=yr)
    
    year_away_offense_eqs = generate_away_park_defense_eqn(
    data = data,  
    batter_data=batter_data,
    league_tbl=league_tbl,
    metric=metric,
    yr=yr)
    print(f"✅ {yr} 年方程式生成完畢：")
    print(f"  park_factor: {len(year_park_eqs)} 條")
    print(f"  home_defense: {len(year_home_defense_eqs)} 條")
    print(f"  away_offense: {len(year_away_offense_eqs)} 條")
    return {
        "park_factor": year_park_eqs,
        "home_defense": year_home_defense_eqs,
        "away_offense": year_away_offense_eqs
    }

year_eqs = collect_eqns(
        data=df,
        park_data=park_df,
        pitch_data=pitch_df,
        batter_data=bat_df,
        league_tbl=league_summary_tbl,
        metric='SLG',
        yr=2024
    )


# eqs_2024 = collect_eqns(
#     data=df,
#     park_data=park_df,
#     pitch_data=pitch_df,
#     batter_data=bat_df,
#     league_tbl=league_summary_tbl,
#     metric='SLG',
#     yr=2024
# )

# dp(eqs_2024)
# park_eqs = generate_park_eqn(
#         data=df,
#         park_data=park_df,
#         league_tbl=get_league_tbl(),
#         metric="SLG",
#         yr=2024)

# home_defense_eqs = generate_home_park_defense_eqn(
#     pitch_data=pitch_df,
#     league_tbl=get_league_tbl(),
#     metric='SLG',
#     yr=2024
# )


# away_offense_eqs = generate_away_park_defense_eqn(
#     data = df,  
#     batter_data=bat_df,
#     league_tbl=league_summary_tbl,
#     metric='SLG',
#     yr=2024
# )
#display_park_equations(park_eqs)
#%%
