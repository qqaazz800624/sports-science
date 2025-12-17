#%%
import pandas as pd
import os
import base64
import plotly.express as px
from team_colors import mlb_colors

# 設定logo size
logo_size_x = 0.4

# 篩選球隊
teams_to_show = []
if teams_to_show:
    plot_df = regression_df[regression_df['Team'].isin(teams_to_show)].copy()
else:
    plot_df = regression_df.copy()


# ---------------------------------------------------------
# 1. 路徑設定 (保持不變)
# ---------------------------------------------------------
LOGOS_DIR = "/Users/yantianli/factor-and-defense-factor/logos"
SOLN_PATH = "/Users/yantianli/factor-and-defense-factor/estimated_factors.csv"

# ---------------------------------------------------------
# 2. 讀取資料與處理 (關鍵修改在這邊!)
# ---------------------------------------------------------
regression_df = pd.read_csv(SOLN_PATH, index_col=None).copy()
regression_df = regression_df[regression_df["Year"] != 2020].copy()

regression_df["Year"] = regression_df["Year"].astype(int)
regression_df["Team"] = regression_df["Team"].str.strip()
regression_df.sort_values(by="Year", inplace=True)


linechart_y = "DefenseFactor"

# ---------------------------------------------------------
# 3. 圖片讀取函式 (加入防止換行符號的處理)
# ---------------------------------------------------------
def get_local_image_b64(team_name):
    file_path = os.path.join(LOGOS_DIR, f"{team_name}.png")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as image_file:
        # 轉 Base64 並移除可能的換行符號，確保字串乾淨
        encoded = base64.b64encode(image_file.read()).decode('utf-8').replace("\n", "").replace("\r", "")
    return f"data:image/png;base64,{encoded}"

# ---------------------------------------------------------
# 4. 繪圖
# ---------------------------------------------------------
fig = px.line(
    plot_df,
    # 因為 Year 已經是數字了，Plotly 會自動使用線性軸
    x="Year", 
    y=linechart_y,
    color="Team",
    title=f"{linechart_y} Trend with Team Logos",
    color_discrete_map=mlb_colors,
    markers=False # 關閉原本的點
)

fig.update_traces(line=dict(width=3))

# ---------------------------------------------------------
# 加入圖片
# ---------------------------------------------------------
y_values = plot_df[linechart_y]
y_range = y_values.max() - y_values.min()

# 設定圖片高度
logo_size_y = y_range * 0.15 if y_range > 0 else 1


image_cache = {}

for index, row in plot_df.iterrows():
    team = row['Team']
    if pd.notna(row[linechart_y]):
        # 讀取與快取
        if team not in image_cache:
            img_source = get_local_image_b64(team)
            image_cache[team] = img_source
        else:
            img_source = image_cache[team]
        
        # 加入圖片
        if img_source:
            fig.add_layout_image(
                dict(
                    source=img_source,
                    xref="x", yref="y",
                    x=row["Year"],      # 直接使用數字年份
                    y=row[linechart_y], # 使用 Y 值
                    sizex=logo_size_x,
                    sizey=logo_size_y,
                    xanchor="center", yanchor="middle",
                    layer="above",
                    sizing="contain"
                )
            )

# ---------------------------------------------------------
# 6. Layout 更新 (優化顯示)
# ---------------------------------------------------------
fig.update_layout(
    xaxis=dict(
        title="Year",
        # 設定刻度格式
        tickmode='linear', # 強制每年顯示一個刻度
        dtick=1,           # 間距為 1 年
        tickformat="d"     # 顯示為整數 (避免出現 2015.5)
    ),
    yaxis=dict(
        title=linechart_y,
        range=[y_values.min() - logo_size_y, y_values.max() + logo_size_y]
    ),
    height=600,
    plot_bgcolor='rgba(250,250,250,1)',
    margin=dict(l=60, r=60, t=80, b=60)
)

# 建議使用瀏覽器開啟，效果最好
fig.show() 
# 如果 renderer="browser" 沒反應，改回 fig.show() 並用 HTML 開啟
#%%