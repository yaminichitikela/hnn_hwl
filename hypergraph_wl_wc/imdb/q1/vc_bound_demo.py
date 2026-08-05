"""
VC-bound non-vacuousness demo -- IMDB datasets (extends Q1's width sweep).

Motivation: the main Q1 width sweep (q1_width_sweep.py) never produces a
non-vacuous VC bound anywhere -- at every width tested (d=4,16,256,1024),
VCdim already exceeds the training-set size m, so the bound is trivially
infinite ("vacuous") and can never meaningfully fail. That is expected
(VC-style bounds scale with the raw parameter count P, and even d=4
already gives P in the hundreds), but it also means the paper's Theorem 2
(thm:vc-hnn) has no empirical footprint anywhere in the main sweep -- the
"VC ok?" column is YES only because the bound is vacuously true, never
because it was actually tested against a real capacity constraint.

This script tests much smaller widths instead, where P shrinks to a
handful of parameters. d=1 is infeasible here: HNNClassifier zero-pads
the raw feat_dim=2 input up to hidden width d, which requires d>=feat_dim
(see the `assert feat_dim <= d` in q1_width_sweep.py's HNNClassifier), so
d=2 (no padding at all) is the smallest width these datasets support.
Tested here: d in {2, 3}. At these widths the bound can come out FINITE
(VCdim < m) on datasets with enough training hypergraphs and a small
enough tuned depth L -- not because the bound becomes tight or
informative in absolute terms (VC-style uniform-convergence bounds are
notoriously loose even when finite; expect triple-digit percentages even
where non-vacuous), but because this is the first and only place in the
whole experimental section where the bound actually responds to
shrinking capacity instead of being vacuous by construction regardless
of what network is tested. Whether d=2/d=3 actually clears the
vacuousness threshold depends on each dataset's own (m, u, L) -- some
will, some won't; report all four here either way.

L and num_epochs are PINNED to the values already found by the main Q1
sweep (job 6435431's logs), not re-tuned here: re-running
tune_hyperparams() would waste real compute re-discovering numbers we
already have (the tuning split is deterministic -- seed=TUNE_SEED -- so
a fresh call would just reproduce the same L and, since every dataset
happened to floor to MIN_REAL_EPOCHS=300 anyway, the same epoch budget).
This script instead reproduces ONLY the deterministic 85%/15% split
tune_hyperparams() depends on (q1.make_split with q1.TUNE_SEED), then
plugs in the already-known (layers, epochs) directly. Every other
function (HWL computation, run_q1, check_q1_bound) is reused unchanged.

Run this yourself (from this directory):
    cd "hypergraph_wl_wc/imdb/q1"
    python3 vc_bound_demo.py
    python3 vc_bound_demo.py --dataset IMDB_dir_genre   # just one dataset
"""
import argparse
import time

import q1_width_sweep as q1

# Small widths only: hunting for a finite (non-vacuous) VC bound, not
# reproducing the main sweep's capacity/generalisation results. d=2 is
# the smallest width feat_dim=2 zero-padding allows; d=3 shows how
# quickly the bound tips back into vacuousness as P grows.
q1.Q1_D_VALUES = [2, 3]

# Pinned from job 6435431's logs (sweep_6435431_0.out) -- every dataset
# happened to floor to MIN_REAL_EPOCHS=300 regardless of its tuned
# estimate, so epochs is a single constant here, not per-dataset.
KNOWN_LAYERS = {
    'IMDB_dir_form': 2,
    'IMDB_dir_genre': 2,
    'IMDB_wri_form': 2,
    'IMDB_wri_genre': 2,
}
KNOWN_EPOCHS = 300

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=q1.IMDB_DATASETS, default=None,
                         help="run only this one dataset instead of all 4")
    args = parser.parse_args()
    datasets_to_run = [args.dataset] if args.dataset else q1.IMDB_DATASETS

    print(f'Device: {q1.DEVICE}  Q1_D_VALUES={q1.Q1_D_VALUES} (VC-bound demo widths)', flush=True)
    for name in datasets_to_run:
        path = q1.IMDB_DIR / f"{name}.txt"
        print(f'\n{"="*70}\n{name}  [VC-bound demo]\n{"="*70}')
        recs, labels, num_classes = q1.load_dataset(path)
        print(f'{len(recs)} graphs, {num_classes} classes, feat_dim={q1.FEAT_DIM}')

        # Reproduces the exact 85% pool tune_hyperparams() would have
        # returned (deterministic split, same TUNE_FRAC/TUNE_SEED),
        # without spending any compute on the tuning-training run itself.
        _, main_recs = q1.make_split(recs, q1.TUNE_FRAC, seed=q1.TUNE_SEED)
        layers, n_epochs = KNOWN_LAYERS[name], KNOWN_EPOCHS
        print(f'  [{name}] using pinned: num_layers={layers} (from job 6435431), '
              f'num_epochs={n_epochs}', flush=True)

        t0 = time.time()
        # Pass 1: standard L=layers rounds (matches the model's depth).
        all_nc, all_ec, hwl_sigs = q1.compute_hwl_init(main_recs)
        for _ in range(layers):
            all_nc, all_ec, hwl_sigs = q1.compute_hwl_round(main_recs, all_nc, all_ec, hwl_sigs)
        # Pass 2: one extra round, continuing from pass 1's state.
        all_nc, all_ec, hwl_sigs = q1.compute_hwl_round(main_recs, all_nc, all_ec, hwl_sigs)
        print(f'HWL computed ({layers} rounds + 1 extra) in {time.time()-t0:.1f}s')

        q1_results = q1.run_q1(main_recs, num_classes, layers, n_epochs, q1.EVAL_EVERY)
        q1.check_q1_bound(name, main_recs, hwl_sigs, q1_results, layers)
