#!/bin/bash
# SLURM submission for the Q4 exact-bound experiment (prop:upper-nonuniform).
# Same conventions as q3_shatter/submit_hpc.sbatch (account "<your-account>",
# partition "gpu", GPU software via the pytorch-gpu container module).
#
# One array task per POOL_SIZE below (order=6, feat_dim=2 fixed), rather
# than per --order as in Q3: a local calibration run showed random
# hypergraphs essentially never collide in HWL signature below pool
# sizes in the ~1000s at these orders (order<=8, pool<=80 -> zero
# natural collisions every time), so the pool size IS the base family
# size here for practical purposes (see data.py's module docstring and
# run_exact_bound_experiment.py's own docstring for the full story,
# including why the augmented family's twin is constructed explicitly
# via make_twin rather than searched for). Sweeping pool size instead of
# order therefore directly sweeps the base family size k -- the more
# informative axis for "how does the required width to shatter scale
# with family size", mirroring Q3's own family-size-vs-width analysis.
#
# Submit with:  sbatch submit_hpc.sbatch

#SBATCH --job-name=hyper_exact_bound
#SBATCH --account=<your-account>
#SBATCH --output=logs/exact_bound_%A_%a.out
#SBATCH --error=logs/exact_bound_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

POOL_SIZES=(20 40 60 100)
POOL_SIZE=${POOL_SIZES[$SLURM_ARRAY_TASK_ID]}

crun.pytorch-gpu python3 run_exact_bound_experiment.py \
    --order 6 \
    --feat_dim 2 \
    --pool_size "$POOL_SIZE" \
    --hidden_dim 8 16 32 64 128 256 \
    --num_assign 50 \
    --train_steps 100000 \
    --lr 1e-3 \
    --seed 6 \
    --file_name "exact_bound_pool${POOL_SIZE}.csv"
