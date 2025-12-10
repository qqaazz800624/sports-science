#%%
import pandas as pd
import numpy as np
import os

from IPython.display import display as dp

import plotly.express as px

import seaborn as sns


pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示




soln_path = "/Users/yantianli/factor-and-defense-factor/estimated_factors.csv"
regression_df = pd.read_csv(soln_path, index_col=None).copy()



# Remove 2020 if present and convert to string for categorical axis (no gap)
regression_df = regression_df[regression_df["Year"] != 2020].copy()
regression_df["Year"] = regression_df["Year"].astype(str)

# MLB Team Colors
mlb_colors = {
# 美國聯盟 (American League)
    'BAL': '#DF4601',  # Baltimore Orioles (Orange)
    'BOS': '#BD3039',  # Boston Red Sox (Red)
    'CWS': '#27251F',  # Chicago White Sox (Black)
    'CLE': '#00385D',  # Cleveland Guardians (Navy)
    'DET': '#0C2340',  # Detroit Tigers (Navy)
    'HOU': '#002D62',  # Houston Astros (Navy)
    'KC':  '#004687',  # Kansas City Royals (Royal Blue)
    'LAA': '#BA0021',  # Los Angeles Angels (Red)
    'MIN': '#002B5C',  # Minnesota Twins (Navy)
    'NYY': '#003087',  # New York Yankees (Navy)
    'OAK': '#003831',  # Oakland Athletics (Forest Green)
    'SEA': '#005C5C',  # Seattle Mariners (Northwest Green/Teal)
    'TB':  '#092C5C',  # Tampa Bay Rays (Navy)
    'TEX': '#003278',  # Texas Rangers (Blue)
    'TOR': '#134A8E',  # Toronto Blue Jays (Blue)

    # 國家聯盟 (National League)
    'AZ': '#A71930',  # Arizona Diamondbacks (Sedona Red)
    'ATL': '#CE1141',  # Atlanta Braves (Scarlet)
    'CHC': '#0E3386',  # Chicago Cubs (Royal Blue)
    'CIN': '#C6011F',  # Cincinnati Reds (Red)
    'COL': '#333366',  # Colorado Rockies (Purple)
    'LAD': '#005A9C',  # Los Angeles Dodgers (Dodger Blue)
    'MIA': '#00A3E0',  # Miami Marlins (Miami Blue)
    'MIL': '#12284B',  # Milwaukee Brewers (Navy)
    'NYM': '#FF5910',  # New York Mets (Blue)
    'PHI': '#BA0C2F',  # Philadelphia Phillies (Red)
    'PIT': '#FDB827',  # Pittsburgh Pirates (Gold) *註：底色通常是黑，但黃色為主視覺亮點
    'SD':  '#FFC425',  # San Diego Padres (Brown)
    'SF':  '#FD5A1E',  # San Francisco Giants (Orange)
    'STL': '#C41E3A',  # St. Louis Cardinals (Red)
    'WSH': '#AB0003',  # Washington Nationals (Red)
}
linechart_y = "DefenseFactor"


# 確保資料依照年份排序，這樣 X 軸才會遞增
regression_df.sort_values(by="Year", inplace=True)

# --- 只顯示特定球隊設定 ---
# 在這裡輸入你想看的球隊代號
teams_to_show = [
    
] 

# 過濾資料
if teams_to_show:
    plot_df = regression_df[regression_df['Team'].isin(teams_to_show)].copy()
else:
    plot_df = regression_df.copy() # 如果是空的，就顯示全部


# 找出資料中的最後一年，用來標示隊名
last_year = plot_df["Year"].max()

# 建立一個 Text 欄位
# 只有當 Year == last_year 時，才填入 Team 名稱，其他填空字串
plot_df['Label'] = plot_df.apply(
    lambda row: row['Team'] if row['Year'] == last_year else "", axis=1
)

fig = px.line(
    plot_df,     
    x="Year",           
    y=linechart_y,     
    color="Team",       # 直接用 Team 分組和上色，不需要 ColorGroup
    title=f"{linechart_y} Trend",
    color_discrete_map=mlb_colors,
    hover_name="Team"
)


# 進階設定：自訂線條與 Marker 樣式
fig.update_traces(
    mode='lines+markers', 
    marker=dict(size=8, line=dict(width=2)) 
)

# 套用雙色效果：讓所有球隊都變成「主色邊框 + 白色填充」
fig.for_each_trace(
    lambda trace: trace.update(
        line=dict(width=3),
        marker=dict(
            color='white',            # 點的中間填成白色
            size=8,
            line=dict(
                color=trace.line.color, # 點的邊框 = 線的顏色
                width=2
            )
        )
    )
)

fig.update_layout(
    xaxis=dict(tickmode='linear', dtick=1), 
    legend_title_text='Team',
)

fig.show()
#%%
