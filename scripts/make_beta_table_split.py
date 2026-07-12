#%%

import pandas as pd
import os
import argparse


def format_cell(beta, se):
    return f"{beta:.4f} ({se:.4f})"


def build_panel_rows(df, beta_col, se_col, years):
    sub = df[df['Year'].isin(years)].copy()
    sub['cell'] = sub.apply(lambda r: format_cell(r[beta_col], r[se_col]), axis=1)
    wide = sub.pivot(index='Team', columns='Year', values='cell').sort_index()
    lines = []
    for team, row in wide.iterrows():
        lines.append(f'{team} & ' + ' & '.join(row[y] for y in years) + r' \\')
    return lines


def to_latex(df, beta_col, se_col, caption, label):
    first_years = [2015, 2016, 2017, 2018, 2019]
    second_years = [2020, 2021, 2022, 2023, 2024]

    lines = []
    lines.append(r'\begin{table*}[htbp]')
    lines.append(r'\small\sf\centering')
    lines.append(f'\\caption{{{caption}\\label{{{label}}}}}')
    lines.append(r'\begin{adjustbox}{max totalsize={\textwidth}{0.92\textheight}}')
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append('Team & ' + ' & '.join(str(y) for y in first_years) + r' \\')
    lines.append(r'\midrule')
    lines.extend(build_panel_rows(df, beta_col, se_col, first_years))
    lines.append(r'\midrule')
    lines.append('Team & ' + ' & '.join(str(y) for y in second_years) + r' \\')
    lines.append(r'\midrule')
    lines.extend(build_panel_rows(df, beta_col, se_col, second_years))
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{adjustbox}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate split (two-panel) beta tables with standard errors")
    parser.add_argument('--input', type=str, default='data/results/test_estimated_factors.csv',
                        help='CSV produced by estimate_factors.py')
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help='Directory to save the .tex files')
    parser.add_argument('--effect', type=str, choices=['park', 'defense', 'both'], default='park',
                        help='Which table to generate')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    specs = {
        'park': dict(
            beta_col='BetaPark', se_col='BetaParkSE',
            caption=('Estimated ballpark effects ($\\tilde{\\beta}^{\\text{park}}_p$), 2015--2024. '
                     'Standard errors, shown in parentheses, are computed as linear contrasts of the '
                     'weighted least-squares coefficient covariance matrix under the centered parameterization.'),
            label='tab:park_beta', filename='table2_park_beta_split.tex'),
        'defense': dict(
            beta_col='BetaDefense', se_col='BetaDefenseSE',
            caption=('Defensive Bases Saved (DBS) ($\\tilde{\\beta}^{\\text{def}}_d$), 2015--2024. '
                     'Standard errors, shown in parentheses, are computed as linear contrasts of the '
                     'weighted least-squares coefficient covariance matrix under the centered parameterization.'),
            label='tab:defense_beta', filename='table6_defense_beta_split.tex'),
    }

    effects = ['park', 'defense'] if args.effect == 'both' else [args.effect]
    for effect in effects:
        spec = specs[effect]
        tex = to_latex(df, spec['beta_col'], spec['se_col'], spec['caption'], spec['label'])
        out_path = os.path.join(args.output_dir, spec['filename'])
        with open(out_path, 'w') as f:
            f.write(tex)
        print(f"Saved {effect} table to {out_path}")


if __name__ == "__main__":
    main()

#%%
