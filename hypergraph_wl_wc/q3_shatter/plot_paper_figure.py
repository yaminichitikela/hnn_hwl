"""Paper-ready Q3 shattering figure, styled after WL-meets-VC's Figure 2
(arXiv 2301.11039, fig:exp1): one line per hidden width d, x-axis the
swept quantity (here hypergraph order n), y-axis the capacity metric
(here % of the 50 random labelings perfectly shattered, matching
tab:q3-results exactly -- NOT the raw mean accuracy
plot_shatter_experiment.py already produces, which is a different,
also-valid diagnostic).

Usage (from this directory):
    python3 plot_paper_figure.py
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WIDTHS = [4, 16, 64, 256]
COLORS = {4: '#1f77b4', 16: '#ff7f0e', 64: '#2ca02c', 256: '#d62728'}
MARKERS = {4: 'o', 16: 's', 64: '^', 256: 'D'}

if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'hyper_tnm_order*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Fraction of the 50 random labelings perfectly shattered (accuracy==1.0),
    # per (order, hidden_dim) -- identical definition to tab:q3-results.
    frac = (
        df.groupby(['order', 'hidden_dim'])['accuracy']
        .apply(lambda s: (s >= 0.999).mean() * 100)
        .reset_index(name='pct_shattered')
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for d in WIDTHS:
        sub = frac[frac['hidden_dim'] == d].sort_values('order')
        ax.plot(sub['order'], sub['pct_shattered'], marker=MARKERS[d],
                color=COLORS[d], label=f'$d={d}$', linewidth=1.8, markersize=5)

    ax.axhline(98, color='grey', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.set_xlabel('Hypergraph order $n$')
    ax.set_ylabel('Labelings shattered (%)')
    ax.set_xticks(sorted(df['order'].unique()))
    ax.set_ylim(-3, 103)
    ax.legend(title='Width', frameon=False)
    fig.tight_layout()

    out_pdf = os.path.join(RESULTS_DIR, 'q3_shatter_fig.pdf')
    out_png = os.path.join(RESULTS_DIR, 'q3_shatter_fig.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_pdf} and {out_png}')
