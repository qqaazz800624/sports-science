#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import standardize_data
import os
from download_team_logos import mlb_colors


data_dir = '/neodata/open_dataset/mlb_data/results'

df_mlb = pd.read_csv(os.path.join(data_dir, 'mlb_defense_factor.csv'))
df_est = pd.read_csv(os.path.join(data_dir, 'estimated_defense_factor.csv'))

df_mlb.rename(columns={df_mlb.columns[0]: 'Team'}, inplace=True)
df_est.rename(columns={df_est.columns[0]: 'Team'}, inplace=True)

df_mlb_std = standardize_data(df_mlb)

#%%

target_teams = ["COL", "CIN", "MIN", "MIL", "PIT"]
plt.figure(figsize=(14, 8))
colors = [mlb_colors[team] for team in target_teams]
years = [int(y) for y in df_mlb.columns[1:]]

for i, team in enumerate(target_teams):
    y_mlb = df_mlb_std[df_mlb_std['Team'] == team].iloc[0, 1:].values.astype(float)
    y_est = df_est[df_est['Team'] == team].iloc[0, 1:].values.astype(float)
    
    color = colors[i]
    
    plt.plot(years, y_est, label=f'{team} (Estimated)', color=color, 
             linestyle='-', linewidth=2.5, marker='o')
    
    plt.plot(years, y_mlb, label=f'{team} (Official)', color=color, 
             linestyle='--', linewidth=1.5, alpha=0.6, marker='x')

    plt.text(years[-1] + 0.1, y_est[-1], team, color=color, 
             fontsize=10, fontweight='bold', va='center')
    plt.text(years[-1] + 0.1, y_mlb[-1], team, color=color, 
             fontsize=10, fontweight='bold', va='center')


plt.axhline(100, color='black', linestyle=':', alpha=0.5, label='League Average (100)')
plt.title('Comparison of Defense Factor Trends: Official vs. Estimated', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Standardized Index (100 + 20z)', fontsize=12)
plt.xticks(years)
plt.ylim(50, 185) 
plt.grid(True, alpha=0.3)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

#%%

save_path = os.path.join(data_dir, 'defense_factor_comparison.png')
plt.savefig(save_path, dpi=300)
print(f"Successfully saved {save_path}")


#%%