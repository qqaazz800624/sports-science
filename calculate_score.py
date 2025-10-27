#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

from expect_score import get_expected_dataset, get_event_distribution, get_whole_dataset



# import fangrph data for batter and p
pitcher_data_fg = pd.read_csv('/Users/yantianli/factor_and_defense_factor/fg_pitcher.csv')
batter_data_fg = pd.read_csv('/Users/yantianli/factor_and_defense_factor/fg_batter.csv')

# add year column
pitcher_data_fg.insert(1, 'year', pd.to_datetime(pitcher_data_fg['game_date'], errors='coerce').dt.year)
batter_data_fg.insert(1, 'year', pd.to_datetime(batter_data_fg['game_date'], errors='coerce').dt.year)


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
        (data['year'] == year) & 
        (data['game_type'] == 'R')
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
                    player_type: str,):
    """
    計算指定球員在指定年度的真實事件分布表。
    回傳：
        DataFrame，欄位 ['events', 'sum_real_count']
    """
    # 篩選出沒有打進場的 data 
    df = data[
        (data['year'] == year) & 
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

cole_nonhip_score = nonhip_score_tbl(get_whole_dataset(),
                                     year=2019,
                                     player_mlbid=543037,
                                     player_type='pitcher')
cole_expected = hip_score_tbl(
    data=get_expected_dataset(),
    dist_df=get_event_distribution(),
    year=2019,
    player_mlbid=543037,
    player_type='pitcher',
    method='expectation',
    n_simulations=1000,
    random_seed = 42
)

display(cole_nonhip_score)
display(cole_expected)
# cole_stimuation = expected_score_tbl(
#     data=get_expected_dataset(),
#     dist_df=get_event_distribution(),
#     year=2019,
#     player_mlbid=543037,
#     player_type='pitcher',
#     method='sampling',
#     n_simulations = 1000,
#     random_seed = 42
# )


#%%
