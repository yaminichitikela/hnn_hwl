"""
Q3 -- shattering experiment, hypergraph analogue of Morris et al.'s
wl_vs_vc/Q3/run_shatter_experiment.py ("WL meet VC", ICML 2023,
arXiv:2301.11039, Sec 5).

Why this exists: Q1/Q2 (q1_width_sweep.py, q2_depth_sweep.py) only check
whether the OBSERVED train/test gap ever exceeds a generalization bound
derived FROM the theorem's VCdim upper bound (L*P*log(2uP), the
delta=1/ReLU case of Theorem 1 in the/hnn_vc_full_proof.tex). On every
dataset tested so far that check is vacuous (VCdim already exceeds the
training-set size m at the SMALLEST width tested), so it can never
meaningfully fail -- it isn't actually testing the theorem, only a loose
downstream consequence of it.

This script instead measures an EMPIRICAL LOWER BOUND on the true VC
dimension directly, the same way Morris et al. do: build a family of
hypergraphs of a given size (data.py's hyper-TNM family, the hypergraph
analogue of their TNM trees), assign RANDOM binary labels to the whole
family, and check whether HNN_slp(d,L) can be trained to fit that random
labeling exactly. Repeated over many random assignments (--num_assign)
and increasing family/graph size (--order), the largest family size that
gets fit under every trial is an empirical VCdim >= that size. THAT number
is what should be compared against the theorem's upper bound -- not the
train/test gap.

Protocol matches wl_vs_vc/Q3/run_shatter_experiment.py: for each order o,
build hyper-TNM family of size floor((o-3)/2)+1, for each of --num_assign
random label draws train a fresh HNN_slp(d,L) from scratch to fit that
exact labeling (early-stopping once every graph in the family is
classified correctly), for each (hidden_dim, num_layers) in the grid.

Run locally (quick smoke test, small order/train_steps to sanity check
the pipeline runs end-to-end):
    cd "hypergraph_wl_wc/q3_shatter"
    python3 run_shatter_experiment.py --order 10 20 --hidden_dim 4 16 \\
        --num_assign 3 --train_steps 2000

Run the real sweep (this is what should go on the HPC cluster -- one
job per --order value via a SLURM array, see submit_hpc.sbatch):
    python3 run_shatter_experiment.py --order 90 --hidden_dim 4 16 64 256 \\
        --num_layers 3 --num_assign 50 --train_steps 100000 \\
        --file_name hyper_tnm_order90.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import get_hyper_tnm_hypergraphs, batch_hypergraphs
from model import HNNClassifier

FEAT_DIM = 1  # constant node feature (x_v = 1.0), matches wl_vs_vc/Q3/data.py's x = ones(order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=[3], nargs='+', help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=[4, 16, 64, 256], nargs='+', help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--order", type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90], nargs='+',
                         help="Graph order |V| (number of nodes) to test")
    parser.add_argument("--num_assign", type=int, default=50, help="Random label assignments to try per order")
    parser.add_argument("--train_steps", type=int, default=100000, help="Max gradient steps per (order, assignment, d, L)")
    parser.add_argument("--file_name", type=str, default="hyper_tnm_results.csv", help="CSV file to store results")
    parser.add_argument("--seed", type=int, default=6, help="Random seed for torch and numpy")
    args = parser.parse_args()
    # hyper-TNM's fixed root+2-hub skeleton needs at least 3 nodes before any
    # leaves are added (data.py's hyper_tnm(0, 0)); an --order below that
    # gives get_hyper_tnm_hypergraphs a negative leaf count, producing an
    # empty family that crashes batch_hypergraphs's max() on an empty
    # sequence instead of a clear error.
    bad_orders = [o for o in args.order if o < 3]
    if bad_orders:
        parser.error(f"--order values must be >= 3 (hyper-TNM needs a root + "
                      f"2 hubs before any leaves); got {bad_orders}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')

    out_dir = Path('./results')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / args.file_name
    results = []

    print(f'Device: {device}  orders={args.order}  hidden_dim={args.hidden_dim}  '
          f'num_layers={args.num_layers}  num_assign={args.num_assign}', flush=True)

    for o in args.order:
        hgs = get_hyper_tnm_hypergraphs(o - 3)
        n = len(hgs)
        x, B, node_mask, he_mask = batch_hypergraphs(hgs)
        print(f'\n{"="*70}\norder={o}  |family|={n}\n{"="*70}', flush=True)

        for a in range(args.num_assign):
            y = torch.randint(0, 2, (n,))

            for h in args.hidden_dim:
                for l in args.num_layers:
                    model = HNNClassifier(feat_dim=FEAT_DIM, d=h, n_layers=l)
                    t0 = time.time()
                    acc = model.fit(
                        x, B, node_mask, he_mask, y,
                        train_steps=args.train_steps, lr=args.lr, device=device,
                        tqdm_prefix=f'o={o},h={h},l={l},a={a}',
                    )
                    elapsed = time.time() - t0
                    results.append({
                        'order': o,
                        'family_size': n,
                        'assignment': a,
                        'hidden_dim': h,
                        'layers': l,
                        'P': model.count_params(),
                        'accuracy': acc,
                        'wall_time_s': elapsed,
                    })
                    pd.DataFrame(results).to_csv(csv_path, index=False)
                    print(f'  o={o} h={h} l={l} a={a+1}/{args.num_assign}: '
                          f'acc={acc:.3f} ({elapsed:.1f}s)', flush=True)

    print(f'\nDone. Results saved to {csv_path}')


if __name__ == '__main__':
    main()
