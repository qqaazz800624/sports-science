import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from typing import Dict
import base64

def add_rtheta_features(df):
   
    mask = (df["description"] == "hit_into_play") & (df["game_type"] == "R")
    work_df = df.loc[mask, ['launch_speed', 'launch_angle']].copy()
    work_df = work_df.dropna(subset=['launch_angle', 'launch_speed'], how='any')
    work_df["launch_speed"] = work_df["launch_speed"].clip(upper=120)

    speed_bins = np.arange(0, 121, 3)
    angle_bins = np.arange(-90, 91, 3)

    work_df["r_bin"] = pd.cut(work_df["launch_speed"], bins=speed_bins, 
                              labels=False, include_lowest=True)
    work_df["theta_bin"] = pd.cut(work_df["launch_angle"], bins=angle_bins, 
                                  labels=False, include_lowest=True)

    work_df["r_theta"] = (
        "r" + work_df["r_bin"].astype(int).astype(str) + 
        "_t" + work_df["theta_bin"].astype(int).astype(str)
    )

    df["r_bin"] = np.nan
    df["theta_bin"] = np.nan
    df["r_theta"] = None 

    df.loc[work_df.index, ["r_bin", "theta_bin", "r_theta"]] = \
        work_df[["r_bin", "theta_bin", "r_theta"]]

    return df


def assign_pitcher_batter_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    is_top = df['inning_topbot'] == 'Top'
    df['pitcher_team'] = np.where(is_top, df['home_team'], df['away_team'])
    df['batter_team'] = np.where(is_top, df['away_team'], df['home_team'])
    
    cols = list(df.columns)
    if 'away_team' in cols:
        insert_pos = cols.index('away_team') + 1
        for col in ['pitcher_team', 'batter_team']:
            if col in cols:
                cols.remove(col)
            cols.insert(insert_pos, col)
            insert_pos += 1
    
    return df[cols]     


class Config:
    def __init__(
        self,
        weights: Dict[str, float] = None,
        data_dir: str = '/neodata/open_dataset/mlb_data/preprocessed',
        filename: str = 'rtheta_prob_tbl.parquet'
    ):
        if weights is None:
            self.weights = {
                'single': 1.0,
                'double': 2.0,
                'triple': 3.0,
                'home_run': 4.0
            }
        else:
            self.weights = weights
            
        self.data_dir = data_dir
        self.filename = filename
        self.file_path = os.path.join(self.data_dir, self.filename)
        

def get_expected_bases_map(config: Config):
                           
    if not os.path.exists(config.file_path):
        raise FileNotFoundError(f"Probability table file not found: {config.file_path}")
    
    prob_df = pd.read_parquet(config.file_path)
    prob_pivot = prob_df.pivot(index='r_theta', columns='events', values='probability').fillna(0)
    
    prob_pivot['expected_bases'] = 0.0
    for event, w in config.weights.items():
        if event in prob_pivot.columns:
            prob_pivot['expected_bases'] += prob_pivot[event] * w
            
    return prob_pivot['expected_bases']

def prepare_regression_data(df: pd.DataFrame, 
                            exp_map: pd.Series,
                            config: Config):
    
    df_bip = df[(df['description'] == 'hit_into_play') &
                (df['game_type'] == 'R')].copy()

    team_mapping = {'ATH': 'OAK'}
    for col in ['home_team', 'away_team', 'batter_team', 'pitcher_team']:
        df_bip[col] = df_bip[col].replace(team_mapping)

    #df_bip['expected_metric'] = df_bip['r_theta'].map(exp_map).fillna(0)
    
    # 修正
    df_bip['expected_metric'] = df_bip['r_theta'].map(exp_map)
    df_bip = df_bip.dropna(subset=['expected_metric'])

    event_weights = config.weights
    df_bip['real_metric'] = df_bip['events'].map(event_weights).fillna(0)

    group_cols = ['game_year', 'home_team', 'pitcher_team']

    agg_df = df_bip.groupby(group_cols).agg({
        'real_metric': 'sum',
        'expected_metric': 'sum',
        'events': 'count' 
    }).reset_index()
    
    agg_df.rename(columns={'events': 'weight', 'real_metric': 'sum_real', 'expected_metric': 'sum_exp'}, inplace=True)
    
    agg_df['avg_residual'] = (agg_df['sum_real'] - agg_df['sum_exp']) / agg_df['weight']
    agg_df['park'] = agg_df['home_team']
    agg_df['defense'] = agg_df['pitcher_team']
    
    return agg_df


def run_year_regression(data, year):
    data_yr = data[data['game_year'] == year].copy()

    model = smf.wls("avg_residual ~ C(park) + C(defense)", data=data_yr, weights=data_yr['weight'])
    res = model.fit()
    
    params = res.params
    all_parks = sorted(data_yr['park'].unique())
    all_defenses = sorted(data_yr['defense'].unique())
    
    beta_park_raw = {} 
    beta_def_raw = {} 
    intercept_raw = params['Intercept']
    
    for p in all_parks:
        key = f"C(park)[T.{p}]"
        beta_park_raw[p] = params.get(key, 0.0)
            
    for d in all_defenses:
        key = f"C(defense)[T.{d}]"
        beta_def_raw[d] = params.get(key, 0.0)
            
    mean_park = np.mean(list(beta_park_raw.values()))
    mean_def = np.mean(list(beta_def_raw.values()))
    
    beta_park_centered = {k: v - mean_park for k, v in beta_park_raw.items()}
    beta_def_centered = {k: v - mean_def for k, v in beta_def_raw.items()}
    adj_intercept = intercept_raw + mean_park + mean_def
    
    std_park = np.std(list(beta_park_centered.values()))
    std_def = np.std(list(beta_def_centered.values()))
    
    beta_park = {}
    for k, v in beta_park_centered.items():
        beta_park[k] = v

    beta_defense = {}
    for k, v in beta_def_centered.items():
        beta_defense[k] = -v

    park_indices = {}
    for k, v in beta_park_centered.items():
        z = v / std_park if std_park > 0 else 0
        park_indices[k] = 100 + 20 * z
        
    defense_indices = {}
    for k, v in beta_def_centered.items():
        #z = v / std_def if std_def > 0 else 0
        z = -v / std_def if std_def > 0 else 0
        #defense_indices[k] = 100 - 20 * z
        defense_indices[k] = 100 + 20 * z
    
    return {
        'year': year,
        'intercept': adj_intercept,
        'beta_park': beta_park,
        'beta_defense': beta_defense,
        'park_factors': park_indices,
        'defense_factors': defense_indices,
        'park_std': std_park, 
        'def_std': std_def
    }

def get_local_image_b64(logos_dir, team_name):
    file_path = os.path.join(logos_dir, f"{team_name}.png")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8').replace("\n", "").replace("\r", "")
    return f"data:image/png;base64,{encoded}"

def standardize_data(df):
    res = df.copy()
    year_cols = df.columns[1:]
    for col in year_cols:
        res[col] = pd.to_numeric(res[col], errors='coerce')
        temp_data = res[~res['Team'].str.lower().isin(['mean', 'std', 'nan'])][col]
        m = temp_data.mean()
        s = temp_data.std()

        res[col] = 100 + 20 * (res[col] - m) / s
    return round(res)