#%%
import pandas as pd
from pybaseball import statcast

from pathlib import Path

year = 2020
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

current_file_path = Path(__file__).resolve()

project_root = current_file_path.parent.parent

save_dir = project_root / "data" / "raw"
print(save_dir)

file_path = save_dir / f"statcast_{year}.csv"
print(f"正在存檔至：{file_path}")
data.to_csv(file_path, index=False)
#%%