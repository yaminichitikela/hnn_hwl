# Q3 -- Shattering Experiment (hypergraph analogue)

Hypergraph analogue of `wl_vs_vc/Q3` (Morris et al., "WL meet VC", ICML
2023). Measures an **empirical lower bound on the true VC dimension** of
`HNN_slp(d, L)` by testing whether it can fit arbitrary random binary
labelings of a synthetic hypergraph family ("hyper-TNM", `data.py`) --
the hypergraph counterpart of their TNM tree family. This is a direct
check of the theorem in `the/hnn_vc_full_proof.tex`, unlike Q1/Q2's
train/test-gap check, which is vacuous on every dataset tested so far and
never actually gets to a decisive answer.

## Files
- `data.py` -- hyper-TNM hypergraph family generator + batching
- `model.py` -- HNN_slp (same architecture as Q1/Q2), single binary output, `fit()` trains to fit a fixed label vector with early stopping
- `run_shatter_experiment.py` -- main sweep over order / hidden_dim / num_layers / random label assignments
- `plot_shatter_experiment.py` -- accuracy-vs-|V| plot, one line per hidden_dim
- `submit_hpc.sbatch` -- SLURM template, configured for a generic HPC cluster (placeholder account, partition `gpu`, one array task per `order` value)

## Quick local smoke test (sanity-check the pipeline runs, not the real result)
```
cd "hypergraph_wl_wc/q3_shatter"
python3 run_shatter_experiment.py --order 10 20 --hidden_dim 4 16 \
    --num_assign 3 --train_steps 2000
```

## Full run on your HPC cluster

1. Copy this folder to your cluster (run from your **local terminal**, not the SSH session):
   ```
   rsync -avz "hypergraph_wl_wc/q3_shatter" <user>@<cluster-hostname>:~/
   ```
2. GPU software may ship as a Singularity/Apptainer container module rather than
   a plain `python3`/`cuda` module pair -- e.g. `module load pytorch-gpu/2.8.0`
   only adding a container path to `PATH`, in which case you run commands
   *inside* the container via a `crun.pytorch-gpu`-style wrapper it provides
   (plain `python3` may still resolve to the login node's bare system Python,
   which has neither torch nor numpy). If `tqdm` is missing from the container
   image, a one-time install into your home directory fixes it:
   ```
   module load pytorch-gpu/2.8.0
   crun.pytorch-gpu python3 -m pip install --user tqdm
   ```
3. Submit: `sbatch submit_hpc.sbatch` -- this launches 9 array tasks (one
   per order in `{10,...,90}`), each sweeping `hidden_dim in {4,16,64,256}`
   x `num_assign=50` x `train_steps=100000`. Requested walltime is 3 days
   per task; watch the first task's actual runtime with `sacct -j <jobid>`
   and shorten/lengthen `--time` for subsequent submissions accordingly --
   larger `order`/`hidden_dim` combos that never shatter will run the full
   `train_steps` before giving up, which is expected (that's the point
   where the true VCdim has been exceeded), not a bug.
4. Once all tasks finish, plot (either on the cluster via `crun.pytorch-gpu
   python3 plot_shatter_experiment.py ...`, or copy the `results/*.csv`
   files back and plot locally):
   ```
   python3 plot_shatter_experiment.py --file_names hyper_tnm_order*.csv
   ```

## Interpreting results
For a given `(order, hidden_dim)`, `accuracy == 1.0` across all
`num_assign` random trials means HNN_slp with that hidden dim shattered a
set of size `family_size = floor((order-3)/2)+1` -- i.e. empirical
`VCdim >= family_size`. The largest `family_size` still shattered as
`order` grows is the number to compare against the theorem's upper bound
`VCdim <= L*P*log(2uP)` (the delta=1/ReLU case) -- if the empirical
lower bound tracks the upper bound's order of magnitude, the theorem is
actually validated, not just "never contradicted."
