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
sample_data = savant_data_14_24.sample(n=2000, random_state=42)

# 可用sample_data to do test
data = sample_data.copy()


df = data.copy()
# 資料整理
sorted_df = df[
    (df["description"] == "hit_into_play") # 球被擊出
    & (df["game_type"] == "R") # regular season
]
#刪除掉沒有 launch_angle 或 launch_speed
sorted_data = sorted_df.dropna(subset=['launch_angle', 'launch_speed'], how='any')
# 把 launch_speed > 120 的值改成 120
sorted_df.loc[sorted_df["launch_speed"] > 120, "launch_speed"] = 120

# 建立顏色對照
unique_events = sorted_df["events"].dropna().unique()
colors = plt.cm.get_cmap("hsv", len(unique_events))
event_to_color = {ev: colors(i) for i, ev in enumerate(unique_events)}
event_to_color["unknown"] = (0.5, 0.5, 0.5, 0.5)  # 為 NaN 預留顏色

# 填補缺值後 map
event_colors = sorted_df["events"].fillna("unknown").map(event_to_color)
event_colors = mcolors.to_rgba_array(event_colors.tolist())

# 繪圖
angles = np.deg2rad(sorted_df["launch_angle"])
radii = sorted_df["launch_speed"]
area = (radii / radii.max()) * 20

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='polar')
sc = ax.scatter(angles, radii, c=event_colors, s=area, alpha=0.7)
ax.set_thetamin(-90)
ax.set_thetamax(90)

# 加上圖例（右側）
handles = [plt.Line2D([0], [0], marker='o', color='w', label=ev,
                      markerfacecolor=event_to_color[ev], markersize=8)
            for ev in event_to_color.keys()]
ax.legend(handles=handles, title='Event', loc='center left', bbox_to_anchor=(1.05, 0.5))

# 加上輔助圓弧線（例：r = 20, 40, 60, 80, 100, 120）
r_lines = np.arange(0, 121, 3)
for r in r_lines:
    ax.plot(np.linspace(np.radians(-90), np.radians(90), 200), 
            [r]*200, '-', color='gray', lw=0.5)

theta_lines = np.arange(-90, 91, 3)
for theta in theta_lines:
    ax.plot(np.radians([theta] * len(radii)), 
            radii, '-', color='gray', lw=0.5)


plt.show()
#%%