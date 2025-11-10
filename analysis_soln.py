#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

soln_path = "/Users/yantianli/factor_and_defense_factor/solution.csv"
soln_df = pd.read_csv(soln_path, index_col=0).copy()


defense_var = soln_df[soln_df["variable"].str.startswith("defense_")]

# --- 解析變數名稱成 team 和 year ---
parts = defense_var["variable"].str.split("_", expand=True)
defense_var["type"] = parts[0]
defense_var["team"] = parts[1]
defense_var["year"] = parts[2].astype(int)


# 每年相對於聯盟平均值去中心化
defense_var["value_centered"] = defense_var["value"] - defense_var.groupby("year")["value"].transform("mean")

# 排序數值大小
year_defense = (
    defense_var[defense_var["year"] == 2024]  # 篩選出 2017 年的資料
    .sort_values(by="value_centered", ascending=True)   # 依照防守值排序
)

dp(year_defense)
# --- 設定要繪製的球隊清單（可自訂，例如: ["NYY", "HOU", "LAD"]；若為空則繪製全部） ---
teams_to_plot = ["HOU", "NYY", "DET", "CLE", "MIL", "CWS"]

# --- 決定要繪圖的資料 ---

if len(teams_to_plot) == 0:
    plot_df = defense_var
else:
    plot_df = defense_var[defense_var["team"].isin(teams_to_plot)]

plot_df = plot_df.sort_values(["team", "year"])

# --- 繪圖 ---
fig = px.line(
    plot_df,
    x="year",
    y="value_centered",
    color="team",
    markers=True,
    title="Defense Factor Trend Relative to Team Average (2015–2024)",
    labels={
        "year": "Year",
        "value_centered": "Defense Factor (Centered)",
        "team": "Team"
    },
)

# --- 美化設定 ---
fig.update_layout(
    width=900,
    height=500,
    template="plotly_white",
    legend_title="Team",
    xaxis=dict(tickmode='linear', tick0=2015, dtick=1),  # 每年顯示一個 tick
)

fig.show()

#%%