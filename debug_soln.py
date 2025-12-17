
import pandas as pd
import plotly.express as px
from team_colors import mlb_colors, mlb_logos

# Mock data based on observed structure
# We'll just load the actual file if possible, or mock it if strictly needed.
# Since we have access to the file, let's load it.

try:
    soln_path = "/Users/yantianli/factor-and-defense-factor/estimated_factors.csv"
    regression_df = pd.read_csv(soln_path, index_col=None).copy()
except Exception as e:
    print(f"Error reading path: {e}")
    exit()

regression_df = regression_df[regression_df["Year"] != 2020].copy()
regression_df["Year"] = regression_df["Year"].astype(str)
regression_df.sort_values(by="Year", inplace=True)

teams_to_show = [] # Show all
if teams_to_show:
    plot_df = regression_df[regression_df['Team'].isin(teams_to_show)].copy()
else:
    plot_df = regression_df.copy()

linechart_y = "DefenseFactor"
last_year = plot_df["Year"].max()

fig = px.line(
    plot_df,     
    x="Year",           
    y=linechart_y,     
    color="Team",      
    title=f"{linechart_y} Trend",
    color_discrete_map=mlb_colors,
    hover_name="Team"
)

y_min = plot_df[linechart_y].min()
y_max = plot_df[linechart_y].max()
y_range = y_max - y_min
if y_range == 0: y_range = 10 
logo_size_y = y_range * 0.15 

print(f"DEBUG: y_min={y_min}, y_max={y_max}, logo_size_y={logo_size_y}")
print(f"DEBUG: First few rows of Year: {plot_df['Year'].head().tolist()}")

image_count = 0
for index, row in plot_df.iterrows():
    team = row['Team']
    if team in mlb_logos:
        image_count += 1
        fig.add_layout_image(
            dict(
                source=mlb_logos[team],
                xref="x",
                yref="y",
                x=row["Year"],
                y=row[linechart_y],
                sizex=0.8, 
                sizey=logo_size_y,
                xanchor="center",
                yanchor="middle",
                layer="above"
            )
        )

print(f"DEBUG: Total images added: {image_count}")
if image_count > 0:
    print("DEBUG: Sample Image Config:")
    print(fig.layout.images[0])
else:
    print("DEBUG: No images added. Checking team names match...")
    print(f"Data Teams: {plot_df['Team'].unique()[:5]}")
    print(f"Logo Keys: {list(mlb_logos.keys())[:5]}")

