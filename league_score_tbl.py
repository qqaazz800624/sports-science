#%%
import pandas as pd
import os

def get_league_tbl():
    path = "/Users/yantianli/factor-and-defense-factor/league_summary_tbl.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback or empty if not found? 
    # But generate_matrix expects it.
    print(f"Warning: {path} not found.") 
    return pd.DataFrame() # Return empty to avoid crash on import, but will crash on usage

#%%