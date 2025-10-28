#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

savant_data_14_24 = pd.read_parquet("/Users/yantianli/factor_and_defense_factor/savant_data_14_24_with_rtheta.parquet")

event_distribution = savant_data_14_24.groupby('r_theta')['events'].value_counts(normalize=True).reset_index()
event_distribution.columns = ['r_theta', 'events', 'probability']

def assign_expected_events(df, dist_df):
    """
    根據 r_theta 的事件機率分佈，對打進場事件 ('hit_into_play') 抽樣 expected_events。
    其他事件保持原始值。
    """
    # 建立分佈字典
    dist_dict = {
        rtheta_value: (sub['events'].values, sub['probability'].values)
        for rtheta_value, sub in dist_df.groupby('r_theta')
    }

    df = df.copy()
    df['expected_events'] = df['events']  # 預設保留原始事件

    # 建立 mask：只針對打進場事件
    mask = (df['description'] == 'hit_into_play') & df['r_theta'].notna()

    # 子集
    df_hit = df.loc[mask]

    # 對每個 r_theta 執行抽樣
    for rtheta_value, (events, probs) in dist_dict.items():
        idx = df_hit.index[df_hit['r_theta'] == rtheta_value]
        if len(idx) > 0:
            chosen = np.random.choice(events, size=len(idx), p=probs / probs.sum())
            df.loc[idx, 'expected_events'] = chosen

    return df


expected_df = assign_expected_events(savant_data_14_24, event_distribution)


cols = ['pitch_type', 'game_date', 'year', 'pitcher', 'batter', 'description',
    'events', 'expected_events', 'launch_speed', 'launch_angle', 'r_theta']

expected_df = assign_expected_events(savant_data_14_24, event_distribution)
expected_df = expected_df[cols]

expected_df.to_parquet("/Users/yantianli/factor_and_defense_factor/savant_data_14_24_with_expected_selected.parquet")
def get_whole_dataset():
    path = "/Users/yantianli/factor_and_defense_factor/savant_data_14_24.parquet"
    return pd.read_parquet(path)

def get_expected_dataset():
    """
    提供給其他專案匯入使用。
    回傳：包含 expected_events 的 DataFrame。
    """
    path = "/Users/yantianli/factor_and_defense_factor/savant_data_14_24_with_expected_selected.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}\n請先執行 expeact_score.py 產生它。")
    return pd.read_parquet(path)

def get_event_distribution():
    """
    提供給其他專案匯入使用。
    回傳：包含 event_distribution 的 DataFrame。
    """
    path = "/Users/yantianli/factor_and_defense_factor/event_distribution.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}\n請先執行 expect_score.py 產生它。")
    return pd.read_parquet(path)

if __name__ == "__main__":
    print("正在建立 expected_events 與 event_distribution 資料集...")

    # 路徑設定
    base_path = "/Users/yantianli/factor_and_defense_factor/savant_data_14_24_with_rtheta.parquet"
    expected_path = "/Users/yantianli/factor_and_defense_factor/savant_data_14_24_with_expected_selected.parquet"
    event_dist_path = "/Users/yantianli/factor_and_defense_factor/event_distribution.parquet"

    # 如果 parquet 檔案已存在，直接略過重算
    if os.path.exists(expected_path) and os.path.exists(event_dist_path):
        print("檔案已存在，略過重算。")
        print(f"檔案位於：\n  - {expected_path}\n  - {event_dist_path}")
    else:
        # 1️⃣ 檢查主資料是否存在
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"主資料集不存在：{base_path}")

        # 2️⃣ 載入主資料
        df = pd.read_parquet(base_path)

        # 3️⃣ 計算 event_distribution
        event_distribution = (
            df.groupby("r_theta")["events"]
            .value_counts(normalize=True)
            .reset_index()
        )
        event_distribution.columns = ["r_theta", "events", "probability"]

        # 4️⃣ 定義抽樣函數（使用改良版）
        def assign_expected_events(df, dist_df):
            dist_dict = {
                rtheta_value: (sub['events'].values, sub['probability'].values)
                for rtheta_value, sub in dist_df.groupby('r_theta')
            }

            df = df.copy()
            df['expected_events'] = df['events']
            mask = (df['description'] == 'hit_into_play') & df['r_theta'].notna()
            df_hit = df.loc[mask]

            for rtheta_value, (events, probs) in dist_dict.items():
                idx = df_hit.index[df_hit['r_theta'] == rtheta_value]
                if len(idx) > 0:
                    chosen = np.random.choice(events, size=len(idx), p=probs / probs.sum())
                    df.loc[idx, 'expected_events'] = chosen
            return df

        # 5️⃣ 執行抽樣 + 欄位選擇
        expected_event_df = assign_expected_events(df, event_distribution)
        cols = [
            'pitch_type', 'game_date', 'year', 'pitcher', 'batter',
            'description', 'events', 'expected_events', 'launch_speed', 'launch_angle', 'r_theta'
        ]
        expected_event_df = expected_event_df[cols]

        # 6️⃣ 儲存結果
        expected_event_df.to_parquet(expected_path)
        event_distribution.to_parquet(event_dist_path)

        print(f"已輸出 expected_event_df 到：{expected_path}")
        print(f"已輸出 event_distribution 到：{event_dist_path}")
#%%