"""
VC-bound non-vacuousness demo -- RHG datasets (extends Q1's width sweep).

See imdb/q1/vc_bound_demo.py for the full rationale (identical here: the
main Q1 sweep's VC bound is vacuous at every tested width, so this script
tests much smaller widths (d=2, d=3 -- d=1 is infeasible, feat_dim=2
zero-padding requires d>=feat_dim) to check whether the bound can ever
come out finite instead). RHG's datasets are tuned to deeper L than IMDB's
(L=4..8 vs L=2), which pushes VCdim up further at fixed d, so these are
a harder case for reaching non-vacuousness -- report all four regardless.

L and num_epochs are PINNED to the values already found by the main Q1
sweep (job 6435431's logs), not re-tuned here -- see
imdb/q1/vc_bound_demo.py's docstring for why. Only the deterministic
85%/15% split tune_hyperparams() depends on is reproduced.

Run this yourself (from this directory):
    cd "hypergraph_wl_wc/rhg/q1"
    python3 vc_bound_demo.py
    python3 vc_bound_demo.py --dataset RHG_pyramid   # just one dataset
"""
import argparse
import time

import q1_width_sweep as q1

# Small widths only: hunting for a finite (non-vacuous) VC bound, not
# reproducing the main sweep's capacity/generalisation results.
q1.Q1_D_VALUES = [2, 3]

# Pinned from job 6435431's logs (sweep_6435431_2.out) -- every dataset
# happened to floor to MIN_REAL_EPOCHS=300 regardless of its tuned
# estimate, so epochs is a single constant here, not per-dataset.
KNOWN_LAYERS = {
    'RHG_pyramid': 4,
    'RHG_10': 8,
    'RHG_3': 7,
    'RHG_table': 8,
}
KNOWN_EPOCHS = 300

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=q1.RHG_DATASETS, default=None,
                         help="run only this one dataset instead of all 4")
    args = parser.parse_args()
    datasets_to_run = [args.dataset] if args.dataset else q1.RHG_DATASETS

    print(f'Device: {q1.DEVICE}  Q1_D_VALUES={q1.Q1_D_VALUES} (VC-bound demo widths)', flush=True)
    for name in datasets_to_run:
        path = q1.RHG_DIR / f"{name}.txt"
        print(f'\n{"="*70}\n{name}  [VC-bound demo]\n{"="*70}')
        recs, labels, num_classes = q1.load_dataset(path)
        print(f'{len(recs)} graphs, {num_classes} classes, feat_dim={q1.FEAT_DIM}')

        _, main_recs = q1.make_split(recs, q1.TUNE_FRAC, seed=q1.TUNE_SEED)
        layers, n_epochs = KNOWN_LAYERS[name], KNOWN_EPOCHS
        print(f'  [{name}] using pinned: num_layers={layers} (from job 6435431), '
              f'num_epochs={n_epochs}', flush=True)

        t0 = time.time()
        all_nc, all_ec, hwl_sigs = q1.compute_hwl_init(main_recs)
        for _ in range(layers):
            all_nc, all_ec, hwl_sigs = q1.compute_hwl_round(main_recs, all_nc, all_ec, hwl_sigs)
        all_nc, all_ec, hwl_sigs = q1.compute_hwl_round(main_recs, all_nc, all_ec, hwl_sigs)
        print(f'HWL computed ({layers} rounds + 1 extra) in {time.time()-t0:.1f}s')

        q1_results = q1.run_q1(main_recs, num_classes, layers, n_epochs, q1.EVAL_EVERY)
        q1.check_q1_bound(name, main_recs, hwl_sigs, q1_results, layers)
