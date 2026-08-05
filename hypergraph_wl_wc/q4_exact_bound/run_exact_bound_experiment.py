"""
Q4 -- exact-bound experiment: empirical test of
Proposition "Upper and matching lower bound" (prop:upper-nonuniform),
VCdim_{H_{d,n}^bool}(HNN_mlp(L)) = m^H_{n,d,L}.

Why this exists: this is the paper's cleanest theoretical result (upper
and lower bound meet exactly, no slack) but, unlike thm:vc-hnn (tested,
if vacuously, in Q1/Q2) and prop:bitlength-exact (tested, if
inconclusively, in Q3), it has NO empirical test anywhere in the
project. This script builds one directly:

  1. Sample a pool of random hypergraphs of a fixed order n, with
     d-dimensional boolean vertex and hyperedge features (data.py).
  2. Run HWL to round L=n (guaranteed stable at that depth for order-n
     hypergraphs). A calibration run showed random pool members
     essentially never collide in HWL signature below pool sizes in the
     ~1000s (order<=8, pool<=80 never produced a single natural
     collision), so a small-to-moderate pool (default 60) is already,
     for practical purposes, itself pairwise HWL-distinguishable -- this
     directly becomes the BASE family, an empirical lower bound on
     m^H_{n,d,L}.
  3. Test: does a sufficiently wide HNN_mlp(L) shatter this exact base
     family? (lower-bound / shattering direction of the theorem.)
  4. Construct one further hypergraph GUARANTEED to share a base
     member's stable signature: an isomorphic copy with vertex labels
     permuted (data.py's make_twin -- HWL is isomorphism-invariant, so
     this works for any n, d, or pool size, unlike waiting for a natural
     collision). Appending it gives the AUGMENTED family (size k+1),
     which pigeonhole guarantees is NOT pairwise distinguishable.
  5. Test: does the fraction of assignments shattered plateau below 1.0
     here, independent of width, for assignments where the twin and its
     partner want different labels -- unlike the base family? (upper-
     bound / pigeonhole direction of the theorem.)

For each width tested, both families get the SAME number of random
binary label assignments (--num_assign), each fit from scratch
(BCEWithLogitsLoss, early-stopping on exact fit) -- same protocol as
q3_shatter/run_shatter_experiment.py's shattering test, extended to two
families and annotated with the twin-vs-partner agreement flag needed to
interpret the augmented family's results.

Run locally (quick smoke test):
    cd "hypergraph_wl_wc/q4_exact_bound"
    python3 run_exact_bound_experiment.py --order 6 --feat_dim 1 \\
        --pool_size 20 --hidden_dim 8 16 --num_assign 5 --train_steps 2000

Run the real experiment (this is what should go on the HPC cluster):
    python3 run_exact_bound_experiment.py --order 6 --feat_dim 2 \\
        --pool_size 60 --hidden_dim 8 16 32 64 128 256 \\
        --num_assign 50 --train_steps 100000 --lr 1e-3 \\
        --file_name exact_bound_n6_d2.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import random_pool, compute_hwl_stable, build_families, make_twin, batch_hypergraphs
from model import HNNmlpClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=6,
                         help="Hypergraph order n (= n_v for every pool member)")
    parser.add_argument("--feat_dim", type=int, default=2,
                         help="Boolean feature dimension d")
    parser.add_argument("--pool_size", type=int, default=60,
                         help="Number of random hypergraphs sampled for the base family. "
                              "Calibration showed collisions are rare below ~1000 at these "
                              "orders, so this size is chosen to be (very likely) already "
                              "pairwise distinguishable on its own, not to search for "
                              "natural collisions -- the augmented family's twin is built "
                              "explicitly (make_twin) regardless of what this pool contains.")
    parser.add_argument("--min_edges", type=int, default=None)
    parser.add_argument("--max_edges", type=int, default=None)
    parser.add_argument("--min_card", type=int, default=2)
    parser.add_argument("--max_card", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=[8, 16, 32, 64, 128, 256], nargs='+',
                         help="HNN_mlp widths to sweep")
    parser.add_argument("--num_layers", type=int, default=None,
                         help="HNN_mlp depth L; defaults to --order (matches the HWL "
                              "rounds needed to guarantee a stable colouring, the depth "
                              "lem:hnn-simulates-hwl's simulation argument needs)")
    parser.add_argument("--hidden_mlp", type=int, default=None,
                         help="Width of the single ReLU hidden layer inside each "
                              "update/readout MLP; defaults to the HNN_mlp width itself")
    parser.add_argument("--num_assign", type=int, default=50,
                         help="Random label assignments to try per (family, width)")
    parser.add_argument("--train_steps", type=int, default=100000,
                         help="Max gradient steps per (family, width, assignment)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--file_name", type=str, default="exact_bound_results.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    L = args.num_layers if args.num_layers is not None else args.order

    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')

    out_dir = Path('./results')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / args.file_name
    summary_path = out_dir / (Path(args.file_name).stem + '_summary.json')

    print(f'Device: {device}  order={args.order}  feat_dim={args.feat_dim}  '
          f'pool_size={args.pool_size}  L={L}  hidden_dim={args.hidden_dim}', flush=True)

    # ---- Step 1-2: sample the pool, run HWL to stability -----------------
    t0 = time.time()
    pool = random_pool(args.order, args.feat_dim, args.pool_size, rng,
                        min_edges=args.min_edges, max_edges=args.max_edges,
                        min_card=args.min_card, max_card=args.max_card)
    all_nc, all_ec = compute_hwl_stable(pool, L)
    print(f'Pool generated and HWL computed ({L} rounds) in {time.time()-t0:.1f}s', flush=True)

    # ---- Step 3: base family (empirical lower bound on m^H_{n,d,L}) ------
    base_indices, natural_twin_idx, natural_twin_partner_pos = build_families(pool, all_nc, all_ec)
    k = len(base_indices)
    print(f'Base family (pairwise HWL-distinguishable): {k} hypergraphs '
          f'out of {args.pool_size} pool members '
          f'(empirical lower bound on m^H_{{n={args.order},d={args.feat_dim},L={L}}})',
          flush=True)
    if natural_twin_idx is not None:
        print(f'  (bonus: pool also contained a natural collision at position '
              f'{natural_twin_partner_pos}, unused below -- the augmented family\'s '
              f'twin is always constructed explicitly, not taken from this.)', flush=True)

    # ---- Step 4: construct the augmented family's twin explicitly --------
    base_hgs = [pool[i] for i in base_indices]
    twin_partner_pos = 0                       # duplicate the first base-family member
    twin_hg = make_twin(base_hgs[twin_partner_pos], rng)
    augmented_hgs = base_hgs + [twin_hg]
    print(f'Augmented family: {k} base members + 1 explicit isomorphic twin '
          f'(duplicates base position {twin_partner_pos}) = {len(augmented_hgs)} '
          f'hypergraphs, guaranteed NOT pairwise distinguishable.', flush=True)

    summary = {
        'order': args.order, 'feat_dim': args.feat_dim, 'pool_size': args.pool_size,
        'num_layers': L, 'base_family_size': k, 'twin_partner_pos': twin_partner_pos,
        'natural_collision_found': natural_twin_idx is not None, 'seed': args.seed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    families = {
        'base': (base_hgs, None),
        'augmented': (augmented_hgs, twin_partner_pos),
    }

    # ---- Step 5: shattering sweep ------------------------------------------
    results = []
    for fam_name, (hgs, partner_pos) in families.items():
        x, B, node_mask, he_mask = batch_hypergraphs(hgs, args.feat_dim)
        n = len(hgs)
        twin_pos = n - 1 if fam_name == 'augmented' else None

        for a in range(args.num_assign):
            y = torch.randint(0, 2, (n,))
            twin_agrees = (bool(y[twin_pos] == y[partner_pos])
                            if twin_pos is not None else None)

            for h in args.hidden_dim:
                model = HNNmlpClassifier(feat_dim=args.feat_dim, d=h, n_layers=L,
                                          hidden=args.hidden_mlp)
                t0 = time.time()
                acc = model.fit(
                    x, B, node_mask, he_mask, y,
                    train_steps=args.train_steps, lr=args.lr, device=device,
                    tqdm_prefix=f'{fam_name},h={h},a={a}',
                )
                elapsed = time.time() - t0
                results.append({
                    'family': fam_name,
                    'family_size': n,
                    'order': args.order,
                    'feat_dim': args.feat_dim,
                    'layers': L,
                    'hidden_dim': h,
                    'P': model.count_params(),
                    'assignment': a,
                    'twin_agrees': twin_agrees,
                    'accuracy': acc,
                    'wall_time_s': elapsed,
                })
                pd.DataFrame(results).to_csv(csv_path, index=False)
                print(f'  {fam_name} h={h} a={a+1}/{args.num_assign}: '
                      f'acc={acc:.3f} twin_agrees={twin_agrees} ({elapsed:.1f}s)', flush=True)

    print(f'\nDone. Results saved to {csv_path}, summary saved to {summary_path}')


if __name__ == '__main__':
    main()
