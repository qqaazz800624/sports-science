#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

import plotly.express as px

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

# years = [y for y in range(2014,2025) if y != 2020]
# base_dir = os.path.join("/", "Users", "yantianli",
#                          "factor-and-defense-factor")

# all_data = []

# # 讀取每年 CSV
# for year in years:
#     file_path = os.path.join(base_dir, f"statcast_{year}.csv")
#     try:
#         df = pd.read_csv(file_path)
#         df['year'] = year  
#         all_data.append(df)
#         print(f"已讀取 {file_path} ({len(df)} 筆)")
#     except FileNotFoundError:
#         print(f"找不到檔案: {file_path}")

# # 5️合併所有資料
# combined_df = pd.concat(all_data, ignore_index=True)
# combined_df.to_csv(os.path.join(base_dir, "statcast_2014_2024.csv"), index=False)

# load data
parquet_path = "savant_data_14_24.parquet"
csv_path = "/Users/yantianli/factor-and-defense-factor/statcast_2014_2024.csv"

# 如果 parquet 已經存在，就直接讀取；否則讀 csv 並轉換成 parquet
if os.path.exists(parquet_path):
    print("讀取Parquet中")
    savant_data_14_24 = pd.read_parquet(parquet_path)
else:
    print("讀取原始 CSV 並轉換中...")
    savant_data_14_24 = pd.read_csv(csv_path)
    savant_data_14_24.to_parquet(parquet_path)
    print("已建立 Parquet 快取檔案。")

print(f"共讀入 {len(savant_data_14_24):,} 筆資料。")

# sample 100比資料的測試檔
sample_data = savant_data_14_24.sample(n=100, random_state=42)

# 可用sample_data to do test
data = savant_data_14_24.copy()


df = data.copy()
# 資料整理
sorted_data = df[
    (df["description"] == "hit_into_play") # 球被擊出
    & (df["game_type"] == "R") # regular season
]
#刪除掉 沒有 launch_angle 或 launch_speed 的 row
sorted_data = sorted_data.dropna(subset=['launch_angle', 'launch_speed'], how='any')

# 將 launch_angle (-90~90) 映射到極座標角度：0° 水平、-90° 向下、90° 向上
#
fig = px.scatter_polar(r=sorted_data["launch_speed"], 
                      theta=sorted_data["launch_angle"],
                      range_theta=[-90,90], 
                      start_angle=0, 
                      direction="counterclockwise",
                      color=sorted_data["events"],)

fig.update_layout(
    width=700,      # 圖寬
    height=1000,    # 圖高（比寬高會讓半圓看起來更直立）
    autosize=False,
    margin=dict(l=40, r=40, t=40, b=40),
)

fig.show()



#%%