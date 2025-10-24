#%%
import pandas as pd
from pybaseball import statcast

import os


data  = pd.read_csv("/Users/yantianli/Downloads/savant_data_2.csv")
print(len(data))
#%%
# pd.set_option('display.max_rows', None)  # 顯示所有行
# pd.set_option('display.max_columns', None)  # 顯示所有列
# pd.set_option('display.width', None)  # 自動調整寬度以適應內容
# pd.set_option('display.max_colwidth', None)  # 不限制單個列的最大寬度
 
year = 2024
  # 可調整年份範圍
# 建立月份區間（每月抓一次）
months = [
    ("03-01", "03-31"),
    ("04-01", "04-30"),
    ("05-01", "05-31"),
    ("06-01", "06-30"),
    ("07-01", "07-31"),
    ("08-01", "08-31"),
    ("09-01", "09-30"),
    ("10-01", "10-31"),
    ("11-01", "11-30"),
]

# 儲存所有月份的資料
all_data = []

for start, end in months:
    print(f"正在抓取 {year}-{start} 到 {year}-{end} ...")
    try:
        df = statcast(start_dt=f"{year}-{start}", end_dt=f"{year}-{end}")
        if df is not None and not df.empty:
            all_data.append(df)
    except Exception as e:
        print(f"{year}-{start} ~ {year}-{end} 抓取失敗：{e}")

# 合併成一個 DataFrame
data = pd.concat(all_data, ignore_index=True)

data.to_csv(f"statcast_{year}.csv", index=False)
#%%