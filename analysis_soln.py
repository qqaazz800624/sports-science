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



def preprocess_factor(df, prefix, year, ascending_way=True):
    """篩選指定類型（如 'park_factor', 'defense_'）並進行解析與去中心化"""
    sub_df = df[df["variable"].str.startswith(prefix)].copy()
    parts = sub_df["variable"].str.split("_", expand=True)

    if prefix == "park_factor":
        sub_df["type"] = parts[0] + "_" + parts[1]
        sub_df["team"] = parts[2]
        sub_df["year"] = parts[3].astype(int)
    else:
        sub_df["type"] = parts[0]
        sub_df["team"] = parts[1]
        sub_df["year"] = parts[2].astype(int)

    # 去中心化（相對於聯盟平均）
    sub_df["value_centered"] = sub_df["value"] - sub_df.groupby("year")["value"].transform("mean")

    # 回傳指定年度排序結果
    year_df = (
        sub_df[sub_df["year"] == year]
        .sort_values(by="value", ascending=ascending_way)
    )
    return sub_df, year_df


soln_path = "/Users/yantianli/factor_and_defense_factor/solution.csv"
soln_df = pd.read_csv(soln_path, index_col=0).copy()

defense_var, year_defense = preprocess_factor(soln_df, "defense_", 2024, ascending_way=True)
park_var, year_park = preprocess_factor(soln_df, "park_factor", 2024, ascending_way=True)

# print(defense_var.sample(5))
dp(year_defense)

fig = px.line(
    year_defense,
    x="team",
    y="value",
    color="team",
    markers=True,
    # title="Park Factor Trend (2015–2024)",
    labels={
        "year": "Year",
        "value_centered": "Defense Factor (Centered)",
        "team": "Team"
    },
)
fig.show()

# --- 設定要繪製的球隊清單（可自訂，例如: ["NYY", "HOU", "LAD"]；若為空則繪製全部） ---
# ['NYM', 'TB', 'CIN', 'AZ', 'LAA', 'TEX', 'CWS', 'DET', 'MIA', 'OAK',
#        'PIT', 'BAL', 'MIL', 'SD', 'TOR', 'COL', 'BOS', 'SEA', 'NYY',
#        'MIN', 'LAD', 'SF', 'PHI', 'STL', 'CHC', 'HOU', 'CLE', 'ATL',
#        'WSH', 'KC']
teams_to_plot = ["HOU", "NYY", "LAD", "LAA", "OAK", "CWS"]

# --- 決定要繪圖的資料 ---
# defense part
if len(teams_to_plot) == 0:
    defense_df = defense_var
else:
    defense_df = defense_var[defense_var["team"].isin(teams_to_plot)]

defense_df = defense_df.sort_values(["team", "year"])
defense_df = (
    defense_df
    .sort_values(["team", "year"])
    .groupby("team", group_keys=False)
    .apply(lambda g: pd.concat([
        g[g["year"] < 2020],
        pd.DataFrame({"year": [2020], "value_centered": [np.nan], "team": [g["team"].iloc[0]]}),
        g[g["year"] > 2020]
    ]))
    .reset_index(drop=True)
)
# park part
if len(teams_to_plot) == 0:
    park_df = park_var
else:
    park_df = park_var[park_var["team"].isin(teams_to_plot)]

park_df = park_df.sort_values(["team", "year"])
park_df = (
    park_df
    .sort_values(["team", "year"])
    .groupby("team", group_keys=False)
    .apply(lambda g: pd.concat([
        g[g["year"] < 2020],
        pd.DataFrame({"year": [2020], "value_centered": [np.nan], "team": [g["team"].iloc[0]]}),
        g[g["year"] > 2020]
    ]))
    .reset_index(drop=True)
)
# --- 繪圖 ---
fig = px.line(
    defense_df,
    x="year",
    y="value_centered",
    color="team",
    markers=True,
    title="Defense Factor Trend (2015–2024)",
    labels={
        "year": "Year",
        "value_centered": "Defense Factor (Centered)",
        "team": "Team"
    },
)


# --- 美化設定 ---
fig.update_layout(
    width=1200,
    height=800,
    template="plotly_white",
    legend_title="Team",
    xaxis=dict(tickmode='linear', tick0=2015, dtick=1),  # 每年顯示一個 tick
)

fig.show()

# --- 繪製 Park Factor 趨勢 ---
fig_park = px.line(
    park_df,
    x="year",
    y="value_centered",
    color="team",
    markers=True,
    title="Park Factor Trend (2015–2024)",
    labels={"year": "Year", "value_centered": "Park Factor (Centered)", "team": "Team"},
)

fig_park.update_layout(
    width=1200,
    height=800,
    template="plotly_white",
    legend_title="Team",
    xaxis=dict(tickmode="linear", tick0=2015, dtick=1),
)
fig_park.show()
#%%