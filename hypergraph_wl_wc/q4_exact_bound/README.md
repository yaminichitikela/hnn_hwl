# Q4 -- Exact-Bound Experiment (prop:upper-nonuniform)

Empirically checks the exact matching upper/lower VC-dimension bound
`VCdim = m^H_{n,d,L}` (`prop:upper-nonuniform`): whether `HNN_mlp(L)` can
shatter exactly as many pairwise-HWL-distinguishable hypergraphs as the
theorem predicts, sweeping the base family size (pool size) rather than
order.

## Files
- `data.py` -- hypergraph pool generator, including the explicit twin
  construction (`make_twin`) used to build the augmented family
- `model.py` -- HNN_mlp architecture used for the shattering test
- `run_exact_bound_experiment.py` -- main sweep over pool size / hidden_dim / random label assignments
- `submit_hpc.sbatch` -- SLURM template, configured for a generic HPC cluster (placeholder account, partition `gpu`, one array task per pool size)
- `requirements.txt` -- Python dependencies

## Quick local smoke test
```
cd "hypergraph_wl_wc/q4_exact_bound"
python3 run_exact_bound_experiment.py --order 6 --feat_dim 2 \
    --pool_size 20 --hidden_dim 8 16 --num_assign 3 --train_steps 2000
```

## Full run on your HPC cluster

1. Copy this folder to your cluster (run from your **local terminal**, not the SSH session):
   ```
   rsync -avz "hypergraph_wl_wc/q4_exact_bound" <user>@<cluster-hostname>:~/hypergraph_wl_wc/
   ```
2. Same container-module conventions as `q3_shatter/submit_hpc.sbatch`
   (see that README for the `crun.pytorch-gpu`-style wrapper details).
3. Submit: `sbatch submit_hpc.sbatch` -- this launches one array task per
   pool size in `{20,40,60,100}`, each sweeping
   `hidden_dim in {8,16,32,64,128,256}` x `num_assign=50` x
   `train_steps=100000`, at fixed `order=6`, `feat_dim=2`.
4. Once all tasks finish, results land in `results/exact_bound_pool<N>.csv`
   for each pool size `<N>`.

## Interpreting results
Sweeping pool size instead of order directly sweeps the base family size
`k`, the more informative axis for "how does the required width to
shatter scale with family size," mirroring Q3's family-size-vs-width
analysis. `accuracy == 1.0` across all `num_assign` trials at a given
`(pool_size, hidden_dim)` means `HNN_mlp` with that hidden dim shattered
the augmented family built from that pool -- an empirical check that the
theorem's exact bound `VCdim = m^H_{n,d,L}` holds in practice, not just
as an upper bound.
