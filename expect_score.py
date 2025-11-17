#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import joblib 

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

# 讀取parquet and cache
parquet_path = "/Users/yantianli/factor_and_defense_factor/truncated_data_with_rtheta.parquet"
cache_path = "/Users/yantianli/factor_and_defense_factor/_cache/savant_data_cache.pkl"

if os.path.exists(cache_path):
    print("從快取讀取主資料中...")
    df = joblib.load(cache_path)
else:
    print("讀取 parquet 並建立快取...")
    df = pd.read_parquet(parquet_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    joblib.dump(df, cache_path)
    print(f"已建立快取：{cache_path}")


# 建立每一個 r theta of  hip 的 events 的 機率 table
rtheta_prob_tbl = df.groupby('r_theta')['events'].value_counts(normalize=True).reset_index()
rtheta_prob_tbl.columns = ['r_theta', 'events', 'probability']
rtheta_prob_tbl.to_parquet("/Users/yantianli/factor_and_defense_factor/rtheta_prob_tbl.parquet")

# 統一路徑設定
BASE_DIR = "/Users/yantianli/factor_and_defense_factor"


# 建立 batter team and pitcher team 的 cols
def assign_pitcher_batter_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    根據每一列的 inning_topbot 欄位，指派 pitcher_team 與 batter_team。

    參數:
    ----------
    df : pd.DataFrame
        原始 Statcast DataFrame，必須包含 'inning_topbot', 'home_team', 'away_team' 欄位。

    回傳:
    ----------
    pd.DataFrame
        加入 'pitcher_team' 和 'batter_team' 欄位的新 DataFrame。
    """
    df = df.copy()
    df['pitcher_team'] = np.where(df['inning_topbot'] == 'Top', df['home_team'], df['away_team'])
    df['batter_team']  = np.where(df['inning_topbot'] == 'Top', df['away_team'], df['home_team'])
    
    # 更改 pitcher team and batter team 的位置
    cols = list(df.columns)
    insert_pos = cols.index('away_team') + 1  # 想放在 away_team 後面
    for col in ['pitcher_team', 'batter_team']:
        cols.remove(col)
        cols.insert(insert_pos, col)
        insert_pos += 1
    df = df[cols]

    return df


def get_whole_dataset():
    """回傳完整的 Parquet 主資料集"""
    path = os.path.join(BASE_DIR, "savant_data_14_24.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到完整資料集：{path}")
    print(f"載入完整資料：{path}")
    return pd.read_parquet(path)

# 讀取特定 cols 的 dataset
def get_truncated_dataset():
    """回傳只含主要欄位的 truncated 資料
    """
    path = os.path.join(BASE_DIR, "truncated_data_with_rtheta.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 truncated 資料：{path}")
    print(f"📦 載入 truncated 資料：{path}")
    return pd.read_parquet(path)

def get_rtheta_prob_tbl():
    """回傳 r_theta 對應的事件機率分佈表"""
    path = os.path.join(BASE_DIR, "rtheta_prob_tbl.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到機率表：{path}\n請先執行 expect_score.py 產生它。"
        )
    print(f"載入 r_theta 機率表：{path}")
    return pd.read_parquet(path)


df = assign_pitcher_batter_teams(get_truncated_dataset())
df.to_parquet("/Users/yantianli/factor_and_defense_factor/truncated_data_with_rtheta_team.parquet")

def get_truncated_dataset_with_team():
    """回傳 tuncated data with rtheta and batter/pitcher team"""
    path = os.path.join(BASE_DIR, "truncated_data_with_rtheta_team.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 truncated 資料：{path}")
    print(f"📦 載入 truncated 資料：{path}")
    return pd.read_parquet(path)
#%%