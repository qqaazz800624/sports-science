#%%
import pandas as pd
import os

from IPython.display import display as dp
import numpy as np

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl, get_whole_dataset
#from team_park_metrics import get_team_score
from league_score_tbl import get_league_tbl

df = get_truncated_dataset_with_team().copy()
dist_df = get_rtheta_prob_tbl()

bat_df = get_team_score("bat")
pitch_df = get_team_score("pitch")
park_df = get_team_score("park")
league_summary_tbl = get_league_tbl()

batter_tm_col = df.pop('batter_team')
pitcher_tm_col = df.pop('pitcher_team')

new_batter_tm_col = df.columns.get_loc('batter') + 1 #type: ignore
new_pitcher_tm_col = df.columns.get_loc('pitcher') + 1 #type: ignore

df.insert(new_batter_tm_col, 'batter_team', batter_tm_col) #type: ignore
df.insert(new_pitcher_tm_col, 'pitcher_team', pitcher_tm_col) #type: ignore


dp(df.head())
#%%