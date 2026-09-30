"""Render the completed OOS report; no model or training dependencies."""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gold_rl_matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.report_dir
    equity = pd.read_csv(root / 'walk_forward_equity.csv', parse_dates=['timestamp'])
    metrics = pd.read_csv(root / 'walk_forward_metrics.csv')
    summary = json.loads((root / 'summary.json').read_text())
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [2, 1]}, constrained_layout=True)
    for name, group in equity.groupby('strategy', sort=False):
        axes[0].plot(group['timestamp'], group['equity'], label=name, linewidth=2.2 if name == 'ppo' else 1.1)
    for fold, group in equity[equity['strategy'] == 'ppo'].groupby('fold'):
        start = group['timestamp'].iloc[0]
        axes[0].axvline(start, color='grey', alpha=.35, linewidth=.8)
        axes[0].text(start, .98, f' Fold {fold}', transform=axes[0].get_xaxis_transform(), va='top', fontsize=9)
    axes[0].axhline(summary['oos_starting_capital'], color='black', linestyle=':', linewidth=.8)
    axes[0].set(title='Walk-forward: equity out-of-sample con capitale continuo', ylabel='Equity')
    axes[0].legend(loc='lower left', ncol=3)
    by_fold = metrics[metrics['scope'] == 'fold'].pivot(index='fold', columns='strategy', values='pnl')
    by_fold.plot.bar(ax=axes[1], width=.8)
    axes[1].set(title='Contributo al PnL per fold — stessi costi e finestre', xlabel='Fold', ylabel='PnL')
    axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].legend(loc='best', ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root / 'walk_forward.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
