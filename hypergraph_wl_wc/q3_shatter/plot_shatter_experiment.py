"""Plot shattering accuracy vs. order, one line per hidden_dim -- same
layout as wl_vs_vc/Q3/plot_shatter_experiment.py. Accuracy=1.0 at a given
(order, hidden_dim) means every random labeling of that family size was
fit exactly, i.e. the empirical VCdim lower bound is at least the
family_size at that order (see run_shatter_experiment.py's docstring).

Usage:
    python3 plot_shatter_experiment.py --file_names hyper_tnm_order10.csv hyper_tnm_order20.csv ...
"""
import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=[3], nargs='+')
    parser.add_argument("--hidden_dim", type=int, default=[4, 16, 64, 256], nargs='+')
    parser.add_argument("--file_names", type=str, nargs='+', default=['hyper_tnm_results.csv'])
    parser.add_argument("--img_dir", type=str, default='results')
    args = parser.parse_args()

    csv_paths = [os.path.join('./results', f) for f in args.file_names]
    df = pd.concat([pd.read_csv(p) for p in csv_paths])
    df['accuracy'] *= 100

    colors = sns.color_palette()
    plt.clf()

    for style_id, h in enumerate(args.hidden_dim):
        h_mask = df['hidden_dim'] == h
        for l in args.num_layers:
            l_mask = df['layers'] == l
            data = df[h_mask & l_mask]
            if data.empty:
                continue

            ax = sns.lineplot(
                x='order', y='accuracy', data=data,
                alpha=1.0, color=colors[style_id], label=f'd={h}',
            )
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.set(xlabel='|V|', ylabel='Shattering accuracy [%]',
                   title='Hyper-TNM shattering (mean over random label assignments)')

    plt.legend()
    os.makedirs(args.img_dir, exist_ok=True)
    plt.savefig(os.path.join(args.img_dir, 'plot_hyper_tnm.png'))
    plt.show()
