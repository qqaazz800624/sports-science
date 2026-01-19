#%%

import pandas as pd
from download_team_logos import mlb_colors
import os
from utils import get_local_image_b64
import plotly.express as px
import argparse

#%%

def main():
    parser = argparse.ArgumentParser(description="Generate Factor Trend Chart with Logos")
    parser.add_argument('--data_dir', type=str, default='/neodata/open_dataset/mlb_data',
                        help='Directory containing MLB data and logos')
    parser.add_argument('--input_filename', type=str, default='results/estimated_factors.csv',
                        help='Filename for the estimated factors CSV')
    parser.add_argument('--teams', nargs='*', type=str, default=[],
                        help='List of team abbreviations to show (default: all teams)')
    parser.add_argument('--save_dir', type=str, default='/neodata/open_dataset/mlb_data/results',
                        help='Directory to save the output chart')
    parser.add_argument('--save_filename', type=str, default='factor_trend_chart.png',
                        help='Filename to save the output chart')
    parser.add_argument('--factor', type=str, default='ParkFactor',
                        help='Factor to plot (options: ParkFactor, DefenseFactor)')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    logos_dir = os.path.join(data_dir, "logos")
    input_filepath = os.path.join(data_dir, args.input_filename)

    regression_df = pd.read_csv(input_filepath, index_col=None)

    regression_df["Year"] = regression_df["Year"].astype(int)
    regression_df["Team"] = regression_df["Team"].str.strip()
    regression_df.sort_values(by="Year", inplace=True)

    linechart_y = args.factor

    if args.teams:
        plot_df = regression_df[regression_df['Team'].isin(args.teams)].copy()
    else:
        plot_df = regression_df.copy()

    fig = px.line(
        plot_df,
        x="Year", 
        y=linechart_y,
        color="Team",
        title=f"{linechart_y} Trend with Team Logos",
        color_discrete_map=mlb_colors,
        markers=False 
    )

    fig.update_traces(line=dict(width=3))

    logo_size_x = 0.4
    y_values = plot_df[linechart_y]
    y_range = y_values.max() - y_values.min()
    logo_size_y = y_range * 0.15 if y_range > 0 else 1

    image_cache = {}

    for index, row in plot_df.iterrows():
        team = row['Team']
        if pd.notna(row[linechart_y]):
            if team not in image_cache:
                img_source = get_local_image_b64(logos_dir=logos_dir, team_name=team)
                image_cache[team] = img_source
            else:
                img_source = image_cache[team]
            
            if img_source:
                fig.add_layout_image(
                    dict(
                        source=img_source,
                        xref="x", yref="y",
                        x=row["Year"],      
                        y=row[linechart_y], 
                        sizex=logo_size_x,
                        sizey=logo_size_y,
                        xanchor="center", yanchor="middle",
                        layer="above",
                        sizing="contain"
                    )
                )

    year_min = plot_df["Year"].min()
    year_max = plot_df["Year"].max()

    fig.update_layout(
        xaxis=dict(
            title="Year",
            tickmode='linear', 
            dtick=1,           
            tickformat="d",
            range=[year_min - 0.5, year_max + 0.5]     
        ),
        yaxis=dict(
            title=linechart_y,
            range=[y_values.min() - logo_size_y, y_values.max() + logo_size_y]
        ),
        height=600,
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(l=60, r=60, t=80, b=60)
    )

    output_path = os.path.join(args.save_dir, args.save_filename)
    fig.write_image(output_path, scale=2)
    print(f"Successfully saved {linechart_y} trend chart to {output_path}")
    
if __name__ == "__main__":
    main()

#%%

# data_dir = "/neodata/open_dataset/mlb_data"
# logos_dir = os.path.join(data_dir, "logos")
# input_filename = "results/estimated_factors.csv"
# input_filepath = os.path.join(data_dir, input_filename)

# regression_df = pd.read_csv(input_filepath, index_col=None)

# regression_df["Year"] = regression_df["Year"].astype(int)
# regression_df["Team"] = regression_df["Team"].str.strip()
# regression_df.sort_values(by="Year", inplace=True)

# linechart_y = "ParkFactor"

# teams_to_show = ["COL", "BOS", "CIN"]
# if teams_to_show:
#     plot_df = regression_df[regression_df['Team'].isin(teams_to_show)].copy()
# else:
#     plot_df = regression_df.copy()

# fig = px.line(
#     plot_df,
#     x="Year", 
#     y=linechart_y,
#     color="Team",
#     title=f"{linechart_y} Trend with Team Logos",
#     color_discrete_map=mlb_colors,
#     markers=False 
# )

# fig.update_traces(line=dict(width=3))

# logo_size_x = 0.4


# y_values = plot_df[linechart_y]
# y_range = y_values.max() - y_values.min()

# logo_size_y = y_range * 0.15 if y_range > 0 else 1

# image_cache = {}

# for index, row in plot_df.iterrows():
#     team = row['Team']
#     if pd.notna(row[linechart_y]):
#         if team not in image_cache:
#             img_source = get_local_image_b64(logos_dir=logos_dir, team_name=team)
#             image_cache[team] = img_source
#         else:
#             img_source = image_cache[team]
        
#         if img_source:
#             fig.add_layout_image(
#                 dict(
#                     source=img_source,
#                     xref="x", yref="y",
#                     x=row["Year"],      
#                     y=row[linechart_y], 
#                     sizex=logo_size_x,
#                     sizey=logo_size_y,
#                     xanchor="center", yanchor="middle",
#                     layer="above",
#                     sizing="contain"
#                 )
#             )

# year_min = plot_df["Year"].min()
# year_max = plot_df["Year"].max()

# fig.update_layout(
#     xaxis=dict(
#         title="Year",
#         tickmode='linear', 
#         dtick=1,           
#         tickformat="d",
#         range=[year_min - 0.5, year_max + 0.5]     
#     ),
#     yaxis=dict(
#         title=linechart_y,
#         range=[y_values.min() - logo_size_y, y_values.max() + logo_size_y]
#     ),
#     height=600,
#     plot_bgcolor='rgba(250,250,250,1)',
#     margin=dict(l=60, r=60, t=80, b=60)
# )

# fig.show() 

#%%    



#%%





#%%