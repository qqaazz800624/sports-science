#%%
import pandas as pd
import os
import joblib

from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示


base_dir = r"/Users/yantianli/factor_and_defense_factor"
# all_data = []

# for year in range(2014, 2025):
#     if year == 2020:  # 跳過2020
#         continue
#     file_path = os.path.join(base_dir, f"statcast_{year}.csv")

#     if os.path.exists(file_path):
#         print(f"讀取中: {file_path}")
#         df = pd.read_csv(file_path)
#         df["year"] = year  # 加上年份欄位以免混淆
#         all_data.append(df)
#     else:
#         print(f"找不到檔案: {file_path}")

# # 合併所有年份資料
# merged_df = pd.concat(all_data, ignore_index=True)

merged_path = os.path.join(base_dir, "savant_data_14_24.parquet")
#merged_df.to_parquet(merged_path)


df = pd.read_parquet(merged_path)


mask = ['pitch_type', 'game_type',
        'game_date', 'game_year', 
        'batter', 'pitcher', 'events', 'description', 
        'inning_topbot', 'home_team', 'away_team',
        'launch_speed', 'launch_angle']
df = df[mask]


df.to_parquet('/Users/yantianli/factor_and_defense_factor/truncated_data.parquet')

def load_savant_data_with_rtheta():
    parquet_path = "/Users/yantianli/factor_and_defense_factor/truncated_data.parquet"
    cache_path = "/Users/yantianli/factor_and_defense_factor/truncated_data.pkl"

    # 若 cache 存在就直接載入
    if os.path.exists(cache_path):
        print("從快取讀取中...")
        df = joblib.load(cache_path)
    else:
        print("讀取 parquet 並建立快取...")
        df = pd.read_parquet(parquet_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump(df, cache_path)
        print(f"已經建立快取檔")
    return df


# judge_df = df[(df['batter'] == 592450)&
#             (df['gayear'] == 2018)]
# print(judge_df['events'].value_counts())


df = load_savant_data_with_rtheta()


# 資料整理
sorted_df = df[
    (df["description"] == "hit_into_play") # 球被擊出
    & (df["game_type"] == "R") # regular season
].copy()
#刪除掉沒有 launch_angle 或 launch_speed
sorted_df = sorted_df.dropna(subset=['launch_angle', 'launch_speed'], how='any')


# 把 launch_speed > 120 的值改成 120
sorted_df.loc[sorted_df["launch_speed"] > 120, "launch_speed"] = 120

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



df.loc[sorted_df.index, ["r_bin", "theta_bin", "r_theta"]] = \
    sorted_df[["r_bin", "theta_bin", "r_theta"]]

output_path = "/Users/yantianli/factor_and_defense_factor/truncated_data_with_rtheta.parquet"
cache_path  = "/Users/yantianli/factor_and_defense_factor/truncated_data_with_rtheta.pkl"


# 輸出 parquet
df.to_parquet(output_path)
print(f"已輸出 parquet：{output_path}")

# 建立快取
joblib.dump(df, cache_path)
print(f"已建立快取檔：{cache_path}")

# 建立顏色對照
unique_events = sorted_df["events"].dropna().unique()
colors = plt.cm.get_cmap("hsv", len(unique_events))
event_to_color = {ev: colors(i) for i, ev in enumerate(unique_events)}
event_to_color["unknown"] = (0.5, 0.5, 0.5, 0.5)  # 為 NaN 預留顏色

# 填補缺值後 map
event_colors = sorted_df["events"].fillna("unknown").map(event_to_color)
event_colors = mcolors.to_rgba_array(event_colors.tolist())

# 移除 unknown 事件
valid_mask = sorted_df["events"].notna()
sorted_df = sorted_df[valid_mask]
event_colors = event_colors[valid_mask]

# 繪圖
angles = np.deg2rad(sorted_df["launch_angle"])
radii = sorted_df["launch_speed"]
area = (radii / radii.max()) * 20

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='polar')
sc = ax.scatter(angles, radii, c=event_colors, s=area, alpha=0.7)
ax.set_thetamin(-90) #type: ignore
ax.set_thetamax(90) #type: ignore

# # 加上圖例（右側）
handles = [plt.Line2D([0], [0], marker='o', color='w', label=ev,
                markerfacecolor=event_to_color[ev], markersize=8)
            for ev in event_to_color.keys() if ev != "unknown"]
ax.legend(handles=handles, title='Event', loc='center left', bbox_to_anchor=(1.05, 0.5))

# 加上輔助圓弧線（例：r = 20, 40, 60, 80, 100, 120）
# 使用既有的 bins
for r in speed_bins:
    ax.plot(np.linspace(np.radians(-90), np.radians(90), 200),
            [r]*200, '-', color='gray', lw=0.5)

for theta in angle_bins:
    ax.plot(np.radians([theta]*200),
            np.linspace(0, 120, 200), '-', color='gray', lw=0.5)


plt.show()
#%%