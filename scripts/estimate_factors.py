#%%

import pandas as pd
import numpy as np
import os
from utils import get_expected_bases_map, prepare_regression_data, run_year_regression, Config
import argparse
from tqdm import tqdm

#%%

def main():
    parser = argparse.ArgumentParser(description="Run Weighted Regression on MLB Data")
    parser.add_argument('--data_dir', type=str, default='/neodata/open_dataset/mlb_data/preprocessed',
                        help='Directory containing preprocessed MLB data')
    parser.add_argument('--input_filename', type=str, default='truncated_data_with_rtheta_team.parquet',
                        help='Filename for the truncated dataset with r_theta and team info')
    parser.add_argument('--prob_table', type=str, default='rtheta_prob_tbl.parquet',
                        help='Filename for the r_theta probability table')
    parser.add_argument('--output_dir', type=str, default='/neodata/open_dataset/mlb_data/results',
                        help='Directory to save regression results')
    parser.add_argument('--output_filename', type=str, default='estimated_factors.csv',
                        help='Filename to save regression results')
    parser.add_argument('--weights', nargs='+', type=float, default=[1.0, 2.0, 3.0, 4.0],
                        help='List of weights for [single, double, triple, home_run]')
    args = parser.parse_args()
    
    current_weights = {
        'single': args.weights[0],
        'double': args.weights[1],
        'triple': args.weights[2],
        'home_run': args.weights[3]
    }

    config = Config(
        weights=current_weights,
        data_dir=args.data_dir,
        filename=args.prob_table
    )

    exp_map = get_expected_bases_map(config=config)
    truncated_file_path = os.path.join(args.data_dir, args.input_filename)
    df = pd.read_parquet(truncated_file_path)
    reg_df = prepare_regression_data(df, exp_map, config)

    years = sorted(reg_df['game_year'].unique())
    results = []

    for year in tqdm(years, desc="Running regressions by year"):
        result = run_year_regression(reg_df, year)
        results.append(result)
    print("Completed regression for all years.")

    output = []
    for result in tqdm(results, desc="Processing results for output"):
        year = result['year']
        teams = set(result['park_factors'].keys()) | set(result['defense_factors'].keys())
        for team in teams:
            output.append({
                'Year': year,
                'Team': team,
                'BetaPark': result['beta_park'].get(team, np.nan),
                'BetaDefense': result['beta_defense'].get(team, np.nan),
                'ParkFactor': result['park_factors'].get(team, np.nan),
                'DefenseFactor': result['defense_factors'].get(team, np.nan),
                'Intercept': result['intercept']
            })
    out_df = pd.DataFrame(output)
    output_path = os.path.join(args.output_dir, args.output_filename)
    out_df.to_csv(output_path, index=False)
    print(f"Successfully saved estimated factors to {output_path}")

if __name__ == "__main__":
    main()

#%%

# import statsmodels.formula.api as smf

# data_dir = '/neodata/open_dataset/mlb_data/preprocessed'
# prob_table_filename = 'rtheta_prob_tbl.parquet'

# current_weights = {
#         'single': 1,
#         'double': 2,
#         'triple': 3,
#         'home_run': 4
#     }

# config = Config(
#         weights=current_weights,
#         data_dir=data_dir,
#         filename=prob_table_filename
#     )

# exp_map = get_expected_bases_map(config=config)

# truncated_filename = 'truncated_data_with_rtheta_team.parquet'
# truncated_file_path = os.path.join(data_dir, truncated_filename)
# df = pd.read_parquet(truncated_file_path)
# reg_df = prepare_regression_data(df, exp_map, config=config)

# year = 2024

# data_yr = reg_df[reg_df['game_year'] == year]
# model = smf.wls("response ~ C(park) + C(defense)", data=data_yr, weights=data_yr['weight'])
# res = model.fit()
# params = res.params

# all_parks = sorted(data_yr['park'].unique())
# all_defenses = sorted(data_yr['defense'].unique())

# beta1 = {} 
# beta2 = {} 

# beta0_raw = params['Intercept']

# # Fill Beta1 (Park)
# # Statsmodels naming: C(park)[T.TeamName]
# for p in all_parks:
#     key = f"C(park)[T.{p}]"
#     if key in params:
#         beta1[p] = params[key]
#     else:
#         beta1[p] = 0.0
        
# # Fill Beta2 (Defense)
# for d in all_defenses:
#     key = f"C(defense)[T.{d}]"
#     if key in params:
#         beta2[d] = params[key]
#     else:
#         beta2[d] = 0.0



# beta2





#%%