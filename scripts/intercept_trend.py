#%%

import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = '/neodata/open_dataset/mlb_data/results'
df = pd.read_csv(os.path.join(data_dir, 'intercept_by_years.csv'))


years = [str(col) for col in df.columns if str(col).isdigit()]
intercepts = df.iloc[0][years].astype(float).values

plt.figure(figsize=(10, 6))

plt.plot(years, intercepts, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('League-Wide Baseline (Intercept) Over Time (2015-2024)', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Intercept Value', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'intercept_trend.png'), dpi=300) 
plt.show()

#%%



