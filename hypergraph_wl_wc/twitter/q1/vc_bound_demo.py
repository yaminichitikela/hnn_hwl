"""
VC-bound non-vacuousness demo -- Twitter-Friend (extends Q1's width sweep).

See imdb/q1/vc_bound_demo.py for the full rationale (identical here: the
main Q1 sweep's VC bound is vacuous at every tested width, so this script
tests much smaller widths (d=2, d=3 -- d=1 is infeasible, feat_dim=2
zero-padding requires d>=feat_dim) to check whether the bound can ever
come out finite instead). Twitter-Friend has p_H=m like Steam-Player, and
also the largest observed empirical gap of any dataset in the whole
project (12.15% at Q2's L=4) -- worth checking whether that is the
dataset where the VC bound, if non-vacuous at all, comes closest to
actually being informative relative to what is really observed.

L and num_epochs are PINNED to the values already found by the main Q1
sweep (job 6435431's logs), not re-tuned here -- see
imdb/q1/vc_bound_demo.py's docstring for why. Only the deterministic
85%/15% split tune_hyperparams() depends on is reproduced.

Run this yourself (from this directory):
    cd "hypergraph_wl_wc/twitter/q1"
    python3 vc_bound_demo.py
"""
import time

import q1_width_sweep as q1

# Small widths only: hunting for a finite (non-vacuous) VC bound, not
# reproducing the main sweep's capacity/generalisation results.
q1.Q1_D_VALUES = [2, 3]

# Pinned from job 6435431's logs (sweep_6435431_6.out).
KNOWN_LAYERS = {'twitter_friend': 2}
KNOWN_EPOCHS = 300

if __name__ == "__main__":
    print(f'Device: {q1.DEVICE}  Q1_D_VALUES={q1.Q1_D_VALUES} (VC-bound demo widths)', flush=True)
    for name in q1.TWITTER_DATASETS:
        path = q1.TWITTER_DIR / f"{name}.txt"
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
