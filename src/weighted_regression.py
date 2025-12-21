#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

from IPython.display import display as dp

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl


prob_df = get_rtheta_prob_tbl()
prob_pivot = prob_df.pivot(index='r_theta', columns='events', values='probability').fillna(0)
weights = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
print(weights.items())

def get_expected_bases_map(metric='slg'):
    """
    Generate a mapping from r_theta bin to Expected Value based on metric.
    metric: 'slg' (Expected Bases), 'avg' (Expected Batting Avg), 'woba' (Linear Weights)
    """
    prob_df = get_rtheta_prob_tbl()
    # Pivot to have events as columns
    # Assuming prob_df has columns: r_theta, events, probability
    prob_pivot = prob_df.pivot(index='r_theta', columns='events', values='probability').fillna(0)
    
    # Define weights for SLG (Total Bases)
    # 1B=1, 2B=2, 3B=3, HR=4
    weights = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    
    # Calculate expected bases
    prob_pivot['expected_bases'] = 0.0
    for event, w in weights.items():
        if event in prob_pivot.columns:
            prob_pivot['expected_bases'] += prob_pivot[event] * w
            
    return prob_pivot['expected_bases']

def prepare_regression_data(df, exp_map):
    """
    Prepare data for weighted regression:
    - Filter for Hit Into Play
    - Calculate Real Total Bases and Expected Total Bases per play
    - Aggregate by Game-Team (unique matchup of Park, Defense, Offense)
    - Calculate Y = log(Real / Expected)
    - Calculate Weight = Count of BIP
    """
    # Filter for hits into play
    # Using description 'hit_into_play' is safer than mapping events
    df_bip = df[df['description'] == 'hit_into_play'].copy()

    # Map expected values
    df_bip['expected_metric'] = df_bip['r_theta'].map(exp_map).fillna(0)

    # Calculate Real Metric (Total Bases for SLG)
    event_weights = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    df_bip['real_metric'] = df_bip['events'].map(event_weights).fillna(0)

    # Aggregate by Game-Team
    # We group by game_year, home_team (Park), pitcher_team (Defense), batter_team
    # We use game_pk if available, else use date/teams as proxy. 
    # The truncated dataset seems not to have game_pk in previous generic inspection, 
    # but grouping by (game_date, home_team, batter_team) is sufficient to identify a game-half.
    # Also include game_year for splitting later.
    
    # Check if 'game_pk' exists
    group_cols = ['game_year', 'home_team', 'pitcher_team', 'batter_team']
    if 'game_pk' in df_bip.columns:
        group_cols.append('game_pk')
    else:
        group_cols.append('game_date') # Proxy for unique game ID

    agg_df = df_bip.groupby(group_cols).agg({
        'real_metric': 'sum',
        'expected_metric': 'sum',
        'events': 'count' # Weight
    }).reset_index()
    
    agg_df.rename(columns={'events': 'weight', 'real_metric': 'sum_real', 'expected_metric': 'sum_exp'}, inplace=True)
    
    # Filter out empty weights or negligible expected values
    # If sum_exp is 0, we cannot estimate a factor.
    # Expected bases for a full game should be >> 1.
    agg_df = agg_df[agg_df['sum_exp'] > 1.0].copy()
    
    # Calculate Y = log(Real / Expected)
    # Use Laplace smoothing (add 0.5) to handle Real=0 (Shutout) and stabilize ratios
    # Real=0, Exp=20 -> log(0.5/20.5) ~ -3.7 (reasonable "bad" game)
    # Real=40, Exp=20 -> log(40.5/20.5) ~ 0.68 (reasonable "good" game)
    agg_df['log_ratio'] = np.log((agg_df['sum_real'] + 0.5) / (agg_df['sum_exp'] + 0.5))
    
    # Define Park and Defense columns explicitly for formula
    agg_df['park'] = agg_df['home_team']
    agg_df['defense'] = agg_df['pitcher_team']
    
    return agg_df

def run_year_regression(data, year):
    """
    Run WLS for a specific year and return adjusted coefficients.
    """
    # Filter by year
    data_yr = data[data['game_year'] == year].copy()
    
    if len(data_yr) < 100:
        print(f"Skipping {year}: Not enough data ({len(data_yr)} rows)")
        return None

    # Fit WLS
    # Model: Y = Beta0 + Park + Defense
    # Statsmodels automatically drops one category (reference) for Park and Defense
    mod = smf.wls("log_ratio ~ C(park) + C(defense)", data=data_yr, weights=data_yr['weight'])
    res = mod.fit()
    
    # Extract and Adjust Coefficients
    params = res.params
    
    all_parks = sorted(data_yr['park'].unique())
    all_defenses = sorted(data_yr['defense'].unique())
    
    # Reconstruct Full Dictionaries
    beta1 = {} # Park Coefficients
    beta2 = {} # Defense Coefficients
    
    beta0_raw = params['Intercept']
    
    # Fill Beta1 (Park)
    # Statsmodels naming: C(park)[T.TeamName]
    for p in all_parks:
        # Check if this park is the reference (omitted) or present
        key = f"C(park)[T.{p}]"
        if key in params:
            beta1[p] = params[key]
        else:
            # This is the Reference Category
            beta1[p] = 0.0
            
    # Fill Beta2 (Defense)
    for d in all_defenses:
        key = f"C(defense)[T.{d}]"
        if key in params:
            beta2[d] = params[key]
        else:
            # Reference
            beta2[d] = 0.0
            
    # --- Adjustment Step (as requested) ---
    # 1. Calculate Average of betas
    mean_beta1 = np.mean(list(beta1.values()))
    mean_beta2 = np.mean(list(beta2.values()))
    
    # 2. Update coeffs: beta_new = beta_old - mean
    beta1_adj = {k: v - mean_beta1 for k, v in beta1.items()}
    beta2_adj = {k: v - mean_beta2 for k, v in beta2.items()}
    
    # 3. Update Intercept: beta0_new = beta0_old + mean1 + mean2
    beta0_adj = beta0_raw + mean_beta1 + mean_beta2
    
    # --- Convert to Factors (100 * exp) ---
    # Park Factor: > 100 means Hitter Friendly (Real > Exp)
    park_factors = {k: 100 * np.exp(v) for k, v in beta1_adj.items()}
    
    # Defense Factor: > 100 means Bad Defense (Real > Exp, i.e., allows more hits)
    # Should check if user wants "Defense Strength" (where > 100 is GOOD).
    # Usually "Factor" on offensive metric implies >100 is "More Offense".
    # So >100 Defense Factor -> More Offense allowed -> Bad Defense.
    # If user wants "Defense Value", might need to invert. 
    # But sticking to "Factor" definition: 100*exp(beta).
    defense_factors = {k: 100 * np.exp(v) for k, v in beta2_adj.items()}
    
    return {
        'year': year,
        'intercept': beta0_adj,
        'park_factors': park_factors,
        'defense_factors': defense_factors
    }

if __name__ == "__main__":
    print("Loading data...")
    df = get_truncated_dataset_with_team()
    exp_map = get_expected_bases_map()
    
    print("Preparing regression data...")
    reg_df = prepare_regression_data(df, exp_map)
    
    years = sorted(reg_df['game_year'].unique())
    results = []
    
    print(f"Running regressions for years: {years}")
    
    for yr in years:
        print(f"--- Processing {yr} ---")
        res = run_year_regression(reg_df, yr)
        if res:
            results.append(res)
            # Print sample to verify
            # print(f"  Intercept: {res['intercept']:.4f}")
            # print(f"  Sample Park (NYY): {res['park_factors'].get('NYY', 'N/A')}")
            
    # Save Results to CSV
    # Create rows: Year, Team, ParkFactor, DefenseFactor
    output_rows = []
    for r in results:
        yr = r['year']
        # Union of teams in park and defense (should be same 30 teams)
        teams = set(r['park_factors'].keys()) | set(r['defense_factors'].keys())
        for tm in teams:
            output_rows.append({
                'Year': yr,
                'Team': tm,
                'ParkFactor': r['park_factors'].get(tm, np.nan),
                'DefenseFactor': r['defense_factors'].get(tm, np.nan),
                'Intercept': r['intercept']
            })
            
    out_df = pd.DataFrame(output_rows)
    save_path = "/Users/yantianli/factor-and-defense-factor/estimated_factors.csv"
    out_df.to_csv(save_path, index=False)
    print(f"Successfully saved estimated factors to {save_path}")
    
    # Display snippet
    dp(out_df.head())
#%%