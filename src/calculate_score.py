#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

from pybaseball import playerid_reverse_lookup, batting_stats, pitching_stats

import joblib


from expect_score import get_whole_dataset, get_truncated_dataset, get_rtheta_prob_tbl

# all_years = []
# for year in range(2014, 2025):
#     print(f"下載 Fangraphs 投手資料：{year}")
#     try:
#         df = pitching_stats(year, qual=0)
#         df["Season"] = year
#         all_years.append(df)
#     except Exception as e:
#         print(f"⚠️ {year} 年資料抓取失敗：{e}")

# pitcher_data_fg = pd.concat(all_years, ignore_index=True)
# pitcher_data_fg.rename(columns={'Season': 'game_year', 'IBB': 'Intent_walk'}, inplace=True)
# pitcher_data_fg= pitcher_data_fg[['IDfg', 'game_year', 'Intent_walk']]
# pitcher_data_fg.to_csv("/Users/yantianli/factor-and-defense-factor/fg_pitching.csv", index=False)

# all_years = []
# for year in range(2014, 2025):
#     print(f"下載 Fangraphs 打者資料：{year}")
#     try:
#         df = batting_stats(year, qual=0)
#         df["Season"] = year
#         all_years.append(df)
#     except Exception as e:
#         print(f"⚠️ {year} 年資料抓取失敗：{e}")

# batter_data_fg = pd.concat(all_years, ignore_index=True)
# batter_data_fg.rename(columns={'Season': 'game_year', 'IBB': 'Intent_walk'}, inplace=True)
# batter_data_fg = batter_data_fg[['IDfg', 'game_year', 'Intent_walk']]
# batter_data_fg.to_csv("/Users/yantianli/factor-and-defense-factor/fg_batting.csv", index=False)


batter_data_fg = pd.read_csv("/Users/yantianli/factor-and-defense-factor/fg_batting.csv")
pitcher_data_fg = pd.read_csv("/Users/yantianli/factor-and-defense-factor/fg_pitching.csv")


def hip_score_tbl(data: pd.DataFrame,
                    dist_df: pd.DataFrame, 
                    year: int, 
                    player_mlbid, 
                    player_type: str,
                    method: str = 'expectation',
                    n_simulations: int= 1000,
                    random_seed: int = 42):
    """
    計算球員的 expected score。
    method 可選：
      - 'expectation'：根據機率加權計算期望值（連續小數）
      - 'sampling'：使用 Monte Carlo 抽樣法（整數次數）
    """

    df = data[
        (data['game_year'] == year) & 
        (data['game_type'] == 'R')&
        (data['description'] == 'hit_into_play')
    ].copy()
    df = df.dropna(subset=['events'])

    if player_mlbid is not None:
        df = df[df[player_type] == player_mlbid]
    
    # real events count
    real_df = df['events'].value_counts().reset_index()
    real_df.columns = ['events', 'sum_real_count']
    
    # r_theta 出現的次數
    counts_dict = df['r_theta'].value_counts().to_dict()
    new_df = dist_df.copy()
    new_df["total_count"] = new_df["r_theta"].map(lambda x: counts_dict.get(x, 0))


    if method == 'expectation':
    # --- 用期望值計算 exptected events ---
        new_df["expected_count"] = new_df["probability"] * new_df["total_count"]

        # normalize 讓期望總數 = 真實總數
        real_event_count = real_df['sum_real_count'].sum()
        expected_total = new_df['expected_count'].sum()
        if expected_total > 0:
            new_df['expected_count'] *= (real_event_count / expected_total)

        new_df['expected_count'] = new_df['expected_count'].round(4)

    elif method == 'sampling':
      # 用抽樣計算 exptected events
      np.random.seed(random_seed)

      # 儲存每次 stimulation 的結果
      simulation_results = []

        
      dist_dict = {
          rtheta: (
              sub['events'].values,
              sub['probability'].values / sub['probability'].sum()
          )
          for rtheta, sub in new_df.groupby('r_theta')
      }

      for sim in range(n_simulations):
        simulated_events = []
        for _, row in df.iterrows():
            rtheta = row['r_theta']
            if rtheta in dist_dict:
                events, probs = dist_dict[rtheta]
                simulated_event = np.random.choice(events, p=probs) # type: ignore
                simulated_events.append(simulated_event)
            else:
                simulated_events.append(np.nan)

        sim_counts = pd.Series(simulated_events).value_counts().rename(f'sim_{sim+1}')
        simulation_results.append(sim_counts)


    # 將所有模擬結果合併成 DataFrame
      sim_df = pd.concat(simulation_results, axis=1).fillna(0)
      sim_df['expected_count'] = sim_df.mean(axis=1)
      new_df = sim_df.reset_index().rename(columns={'index': 'events'})

    else:
        raise ValueError("method 必須是 'expectation' 或 'sampling'")

    # ---- 合併結果 ----
    expected_df = (
        new_df.groupby('events')['expected_count']
        .sum()
        .reset_index()
        .rename(columns={'expected_count': 'sum_expected_count'})
    )

    combined = pd.merge(expected_df, real_df, on='events', how='outer').fillna(0)
    combined = combined.sort_values('sum_real_count', ascending=False).reset_index(drop=True)

    return combined


def nonhip_score_tbl(data: pd.DataFrame, 
                    year: int, 
                    player_mlbid, 
                    player_type: str):
    """
    計算指定球員在指定年度的真實事件分布表。
    回傳：
        DataFrame，欄位 ['events', 'sum_real_count']
    """
    # 篩選出沒有打進場的 data 
    df = data[
        (data['game_year'] == year) & 
        (data['game_type'] == 'R') &
        (data['description'] != 'hit_into_play')
    ].copy()
    df = df.dropna(subset=['events'])

    if player_mlbid is not None:
        df = df[df[player_type] == player_mlbid]

    # 統計每個事件的次數
    real_df = df['events'].value_counts().reset_index()
    real_df.columns = ['events', 'sum_real_count']
    real_df['sum_expected_count'] = real_df['sum_real_count']

    # 依照次數排序
    real_df = real_df.sort_values('sum_real_count', ascending=False).reset_index(drop=True)

    return real_df



def ibb_score_tbl(year: int, 
                player_mlbid: int, 
                player_type: str,
                pitcher_data: pd.DataFrame = pitcher_data_fg,
                batter_data: pd.DataFrame = batter_data_fg) -> int:
    """
    根據 Fangraphs 資料取得指定球員在指定年份的 IBB 數量。
    假設 MLBAM ID 可透過 reverse_lookup 轉為 Fangraphs ID。
    """
    if player_type == 'pitcher':
        df = pitcher_data
    elif player_type == 'batter':
        df = batter_data
    else:
        raise ValueError("player_type 必須是 'pitcher' 或 'batter'")
    
    player_mlbid = int(player_mlbid)

    # 將 savant 的 player id 轉成 fangraphs
    lookup_fgid = playerid_reverse_lookup([player_mlbid], key_type='mlbam')

    # playerid_reverse_lookup([player_mlbid], key_type='mlbam')\
    #     ['key_fangraphs'].values[0]
    if not isinstance(lookup_fgid, pd.DataFrame) or lookup_fgid.empty or 'key_fangraphs' not in lookup_fgid.columns:
        print(f"⚠️ 無法找到 MLB ID {player_mlbid} 對應的 Fangraphs ID，IBB 設為 0。")
        return 0

    fg_id = lookup_fgid['key_fangraphs'].values[0]

    # 篩選選手的資料
    player_data = df[(df['game_year'] == year) &
                    (df['IDfg'] == fg_id)]
    # 回傳 IBB 總數
    if not player_data.empty and 'IBB' in player_data.columns:
        return int(player_data['IBB'].sum())
    else:

        return 0
    

def combined_score_tbl(data: pd.DataFrame,
                    dist_df: pd.DataFrame,
                    year: int,
                    player_mlbid: int,
                    player_type: str,
                    method: str = 'expectation',
                    pitcher_data: pd.DataFrame = pitcher_data_fg,
                    batter_data: pd.DataFrame = batter_data_fg):
    """
    結合：
    - hip_score_tbl（打進場預期與實際）
    - nonhip_score_tbl（非打進場事件）
    - ibb_value_tbl（Fangraphs 的 IBB 資料）

    回傳：包含所有事件的完整統計表。
    """
    # calculate hip events
    hip_df = hip_score_tbl(
        data=data,
        dist_df=dist_df,
        year=year,
        player_mlbid=player_mlbid,
        player_type=player_type,
        method=method
    )

    # calculte nonhip events
    nonhip_df = nonhip_score_tbl(
        data=data,
        year=year,
        player_mlbid=player_mlbid,
        player_type=player_type
    )
    # add up IBB
    ibb_value = ibb_score_tbl(year, 
                            player_mlbid, 
                            player_type,
                            pitcher_data = pitcher_data_fg,
                            batter_data = batter_data_fg)
    ibb_value_df = pd.DataFrame(
        [
            {
                'events': 'intent_walk',
                'sum_real_count': ibb_value,
                'sum_expected_count': ibb_value
            }
        ]
    )
    # combin whole score
    
    combined_df = pd.concat([hip_df, nonhip_df, ibb_value_df], ignore_index=True)
    combined_df = combined_df.groupby("events", as_index=False)\
        [["sum_real_count", "sum_expected_count"]].sum()
    combined_df = combined_df.sort_values('events', ascending=True).reset_index(drop=True)
    return combined_df







# cole_score = combined_score_tbl(
#     data=get_truncated_dataset(),
#     dist_df=get_rtheta_prob_tbl(),
#     year=2022,
#     player_mlbid=592450,  # Judge
#     player_type='batter',
#     method='expectation'
# )

# display(cole_score)
#%%
