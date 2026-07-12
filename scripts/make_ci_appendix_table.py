#%%

import pandas as pd
import numpy as np
import os
import argparse


def build_table(df, factor_col, se_col, point_decimals=0, ci_decimals=1):
    """依十年平均排序，輸出 point (lower, upper) 格式的寬表"""
    df = df.copy()
    df['lower'] = df[factor_col] - 1.96 * df[se_col]
    df['upper'] = df[factor_col] + 1.96 * df[se_col]

    point_fmt = f"{{:.{point_decimals}f}}"
    ci_fmt = f"{{:.{ci_decimals}f}}"
    df['cell'] = (df[factor_col].map(point_fmt.format)
                  + ' (' + df['lower'].map(ci_fmt.format)
                  + ', ' + df['upper'].map(ci_fmt.format) + ')')

    wide = df.pivot(index='Team', columns='Year', values='cell')
    avg = df.groupby('Team')[factor_col].mean()
    wide['Avg'] = avg.map(point_fmt.format)
    wide = wide.loc[avg.sort_values(ascending=False).index]
    return wide


def to_latex(wide, caption, label):
    years = [c for c in wide.columns if c != 'Avg']
    col_spec = 'l' + 'r' * len(wide.columns)
    lines = []
    lines.append(r'\begin{sidewaystable*}')
    lines.append(r'\small\sf\centering')
    lines.append(f'\\caption{{{caption}\\label{{{label}}}}}')
    lines.append(r'\begin{adjustbox}{max width=\textheight}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')
    header = 'Team & ' + ' & '.join(str(c) for c in wide.columns) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    for team, row in wide.iterrows():
        lines.append(f'{team} & ' + ' & '.join(row.values) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{adjustbox}')
    lines.append(r'\end{sidewaystable*}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate appendix tables of standardized effects with 95% CIs")
    parser.add_argument('--input', type=str, default='data/results/test_estimated_factors.csv',
                        help='CSV produced by estimate_factors.py')
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help='Directory to save the .tex files')
    parser.add_argument('--effect', type=str, choices=['park', 'defense', 'both'], default='park',
                        help='Which table to generate')
    parser.add_argument('--point_decimals', type=int, default=0,
                        help='Decimal places for the point estimate and Avg column')
    parser.add_argument('--ci_decimals', type=int, default=1,
                        help='Decimal places for the CI bounds')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    specs = {
        'park': dict(
            factor_col='ParkFactor', se_col='ParkFactorSE',
            caption=('Standardized ballpark effects with 95\\% confidence intervals, 2015--2024. '
                     'Each entry reports the standardized index defined in equation~\\eqref{eq:index} '
                     'with its 95\\% confidence interval in parentheses. '
                     'Rows are ordered by the ten-year average, from most to least conducive to total bases.'),
            label='tab:park_ci', filename='table_E1_park_ci.tex'),
        'defense': dict(
            factor_col='DefenseFactor', se_col='DefenseFactorSE',
            caption=('Standardized team-defense effects with 95\\% confidence intervals, 2015--2024. '
                     'Each entry reports the standardized index defined in equation~\\eqref{eq:index} '
                     'with its 95\\% confidence interval in parentheses. '
                     'Rows are ordered by the ten-year average, from strongest to weakest team defense.'),
            label='tab:defense_ci', filename='table_F1_defense_ci.tex'),
    }

    effects = ['park', 'defense'] if args.effect == 'both' else [args.effect]
    for effect in effects:
        spec = specs[effect]
        wide = build_table(df, spec['factor_col'], spec['se_col'],
                           point_decimals=args.point_decimals, ci_decimals=args.ci_decimals)
        tex = to_latex(wide, spec['caption'], spec['label'])
        out_path = os.path.join(args.output_dir, spec['filename'])
        with open(out_path, 'w') as f:
            f.write(tex)
        print(f"Saved {effect} table to {out_path}")


if __name__ == "__main__":
    main()

#%%
