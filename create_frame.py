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


# load data
parquet_path = "savant_data_14_24.parquet"
csv_path = "//Users/yantianli/factor_and_defense_factor/savant_14_24.csv"

# 如果 parquet 已經存在，就直接讀取；否則讀 csv 並轉換成 parquet
if "savant_data_14_24" not in globals():
    parquet_path = "savant_data_14_24.parquet"
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
data = savant_data_14_24.copy()

df = data.copy()
# 資料整理
sorted_df = df[
    (df["description"] == "hit_into_play") # 球被擊出
    & (df["game_type"] == "R") # regular season
].copy()
#刪除掉沒有 launch_angle 或 launch_speed
sorted_df = sorted_df.dropna(subset=['launch_angle', 'launch_speed'], how='any')
# 把 launch_speed > 120 的值改成 120
sorted_df.loc[sorted_df["launch_speed"] > 120, "launch_speed"] = 120

#
# 設定格線區間
speed_bins = np.arange(0, 121, 3)  # 0 到 120，每3單位一格
angle_bins = np.arange(-90, 91, 3)  # -90 到 90，每3單位一格

# 分箱
sorted_df["r_bin"] = pd.cut(sorted_df["launch_speed"], bins=speed_bins, # type: ignore
                            labels=False, include_lowest=True) # type: ignore
sorted_df["theta_bin"] = pd.cut(sorted_df["launch_angle"], bins=angle_bins, # type: ignore
                                labels=False, include_lowest=True) # type: ignore

# 合併成 r_theta 標籤
sorted_df["r_theta"] = "r" + sorted_df["r_bin"].astype(str) + "_t" + sorted_df["theta_bin"].astype(str)



savant_data_14_24.loc[sorted_df.index, ["r_bin", "theta_bin", "r_theta"]] = \
    sorted_df[["r_bin", "theta_bin", "r_theta"]]


savant_data_14_24.to_parquet("savant_data_14_24_with_rtheta.parquet")
#%%
# # 建立顏色對照
# unique_events = sorted_df["events"].dropna().unique()
# colors = plt.cm.get_cmap("hsv", len(unique_events))
# event_to_color = {ev: colors(i) for i, ev in enumerate(unique_events)}
# event_to_color["unknown"] = (0.5, 0.5, 0.5, 0.5)  # 為 NaN 預留顏色

# # # 填補缺值後 map
# event_colors = sorted_df["events"].fillna("unknown").map(event_to_color)
# event_colors = mcolors.to_rgba_array(event_colors.tolist())

# # # 繪圖
# angles = np.deg2rad(sorted_df["launch_angle"])
# radii = sorted_df["launch_speed"]
# area = (radii / radii.max()) * 20

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(projection='polar')
# sc = ax.scatter(angles, radii, c=event_colors, s=area, alpha=0.7)
# ax.set_thetamin(-90) #type: ignore
# ax.set_thetamax(90) #type: ignore

# # # 加上圖例（右側）
# handles = [plt.Line2D([0], [0], marker='o', color='w', label=ev,
#                       markerfacecolor=event_to_color[ev], markersize=8)
#             for ev in event_to_color.keys()]
# ax.legend(handles=handles, title='Event', loc='center left', bbox_to_anchor=(1.05, 0.5))

# # 加上輔助圓弧線（例：r = 20, 40, 60, 80, 100, 120）
# # 使用既有的 bins
# for r in speed_bins:
#     ax.plot(np.linspace(np.radians(-90), np.radians(90), 200),
#             [r]*200, '-', color='gray', lw=0.5)

# for theta in angle_bins:
#     ax.plot(np.radians([theta]*200),
#             np.linspace(0, 120, 200), '-', color='gray', lw=0.5)


# plt.show()
#%%