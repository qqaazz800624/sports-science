#%%
import pandas as pd

import pandas as pd
import numpy as np

from pathlib import Path

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"

# check if folder exists
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# path of merged data
merged_path = DATA_PROCESSED / "savant_data_14_24.parquet"

if merged_path.exists():
    print(f"已存在合併檔: {merged_path}, 直接讀取")
    df = pd.read_parquet(merged_path)
else: 
    print(f"merge data not found, start to merge")
    all_data = []

    for year in range(2014, 2025):
        file_path = DATA_RAW / f"statcast_{year}.csv"

        if file_path.exists():
            print(f"讀取中: {file_path}")
            df = pd.read_csv(file_path)
            df["year"] = year  # 加上年份欄位以免混淆
            all_data.append(df)
        else:
            print(f"找不到檔案: {file_path}")

    # 合併所有年份資料
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)

        # 存到 processed 資料夾
        merged_path = DATA_PROCESSED / "savant_data_14_24.parquet"
        
        print(f"正在寫入合併檔: {merged_path}")
        merged_df.to_parquet(merged_path)
    else:
        print("沒有讀到任何資料, please check folder data/raw ")



df = pd.read_parquet(merged_path)


mask = ['pitch_type', 'game_type',
        'game_date', 'game_year', 
        'batter', 'pitcher', 'events', 'description', 
        'inning_topbot', 'home_team', 'away_team',
        'launch_speed', 'launch_angle']
df = df[mask]


df.to_parquet(DATA_PROCESSED / "truncated_data.parquet")




df = pd.read_parquet(DATA_PROCESSED / "truncated_data.parquet")

# judge_df = df[(df['batter'] == 592450)&
#             (df['game_year'] == 2018)]
# print(judge_df['events'].value_counts())
def add_rtheta_features(df):
    """
    計算 launch_speed 和 launch_angle 的分箱，並新增 r_theta 欄位。
    注意：只有 'hit_into_play' 且 'Regular Season' 的資料會有值，
    其他的列在這些欄位會是 NaN。
    """
    print("正在計算 radius-theta binning...")

    mask = (df["description"] == "hit_into_play") & (df["game_type"] == "R")
    
    # 取出要處理的子集
    work_df = df.loc[mask, ['launch_speed', 'launch_angle']].copy()
    
    # 刪除缺失值 (這會導致 work_df 的列數變少)
    work_df = work_df.dropna(subset=['launch_angle', 'launch_speed'], how='any')

    # --- 資料處理邏輯 ---
    
    # 把 launch_speed > 120 的值改成 120 (Clip)
    work_df["launch_speed"] = work_df["launch_speed"].clip(upper=120)

    # 設定格線區間
    speed_bins = np.arange(0, 121, 3)
    angle_bins = np.arange(-90, 91, 3)

    # 分箱 (Binning)
    work_df["r_bin"] = pd.cut(work_df["launch_speed"], bins=speed_bins, 
                              labels=False, include_lowest=True)
    work_df["theta_bin"] = pd.cut(work_df["launch_angle"], bins=angle_bins, 
                                  labels=False, include_lowest=True)

    # 合併成 r_theta 標籤 (例如: r20_t30)
    work_df["r_theta"] = (
        "r" + work_df["r_bin"].astype(int).astype(str) + 
        "_t" + work_df["theta_bin"].astype(int).astype(str)
    )

    # --- 關鍵：將計算結果合併回原始的大表 df ---
    # 因為 work_df 是 df 的子集，我們利用 index 來對齊
    print("combining features...")
    
    # 預先建立欄位 (避免 Pandas 碎片化警告)
    df["r_bin"] = np.nan
    df["theta_bin"] = np.nan
    df["r_theta"] = None # 字串欄位通常用 None 初始化

    # 利用 index 更新數值
    df.loc[work_df.index, ["r_bin", "theta_bin", "r_theta"]] = \
        work_df[["r_bin", "theta_bin", "r_theta"]]

    return df

# --- 執行區塊 ---
if __name__ == "__main__":
    # 1. 讀取 
    input_path = DATA_PROCESSED / "truncated_data.parquet"
    
    if input_path.exists():
        df = pd.read_parquet(input_path)
        
        # 2. 處理特徵
        df = add_rtheta_features(df)
        
        # 3. 存檔 (Pathlib 風格)
        output_parquet = DATA_PROCESSED / "truncated_data_with_rtheta.parquet"
        
        print(f"儲存 Parquet: {output_parquet}")
        df.to_parquet(output_parquet)
        
        # (選用) Pickle 存檔
        # cache_path = DATA_PROCESSED / "truncated_data_with_rtheta.pkl"
        # joblib.dump(df, cache_path)
        # print(f"💾 儲存 Pickle: {cache_path}")
        
    else:
        print(f"找不到輸入檔: {input_path}")

#%%
