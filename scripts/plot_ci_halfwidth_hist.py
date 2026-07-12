#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def draw_hist(ax, halfwidths, title, bins, xlim):
    ax.hist(halfwidths, bins=bins, color='#A8BEE8', edgecolor='white', linewidth=0.8, zorder=3)

    mean = halfwidths.mean()
    std = halfwidths.std()
    ax.axvline(mean, color='#1F3A6E', linestyle='--', linewidth=1.5, zorder=4)
    ax.text(mean + 0.4, ax.get_ylim()[1] * 0.95,
            f'mean = {mean:.2f}\nstd = {std:.2f}',
            color='#1F3A6E', fontsize=10, va='top')

    ax.set_xlim(xlim)
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    return mean, std


def plot_halfwidth_hist(halfwidths, title, output_path, bins, xlim):
    fig, ax = plt.subplots(figsize=(6, 4))
    mean, std = draw_hist(ax, halfwidths, title, bins, xlim)
    ax.set_xlabel('95% CI half-width (index scale)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return mean, std


def plot_combined(park_hw, def_hw, output_path, bins, xlim):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    draw_hist(axes[0], park_hw, '(a) Ballpark effects', bins, xlim)
    draw_hist(axes[1], def_hw, '(b) Team-defense effects', bins, xlim)
    axes[1].set_xlabel('95% CI half-width (index scale)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot histograms of 95% CI half-widths on the index scale")
    parser.add_argument('--input', type=str, default='data/results/test_estimated_factors.csv',
                        help='CSV produced by estimate_factors.py')
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help='Directory to save the figures')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    park_hw = 1.96 * df['ParkFactorSE']
    def_hw = 1.96 * df['DefenseFactorSE']

    # 兩張圖共用同一組 bins 與 x 軸範圍，方便對照
    lo = np.floor(min(park_hw.min(), def_hw.min()))
    hi = np.ceil(max(park_hw.max(), def_hw.max()))
    bins = np.arange(lo, hi + 2, 2)
    xlim = (lo - 1, hi + 1)

    stats = {}
    stats['park'] = plot_halfwidth_hist(
        park_hw, 'Ballpark effects: distribution of 95% CI half-widths (n = 300)',
        os.path.join(args.output_dir, 'park_ci_halfwidth_hist.png'), bins, xlim)
    stats['defense'] = plot_halfwidth_hist(
        def_hw, 'Team-defense effects: distribution of 95% CI half-widths (n = 300)',
        os.path.join(args.output_dir, 'defense_ci_halfwidth_hist.png'), bins, xlim)
    plot_combined(park_hw, def_hw,
                  os.path.join(args.output_dir, 'ci_halfwidth_hist_combined.png'), bins, xlim)

    for effect, (mean, std) in stats.items():
        print(f"{effect}: mean = {mean:.2f}, std = {std:.2f}")


if __name__ == "__main__":
    main()

#%%
