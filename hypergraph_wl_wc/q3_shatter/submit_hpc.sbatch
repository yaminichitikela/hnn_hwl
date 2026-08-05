#!/bin/bash
# SLURM submission template for the hypergraph shattering (Q3) sweep.
# Configured for our institution's HPC cluster (verified interactively on a login node):
#   - account "<your-account>", partition "gpu" (16 nodes, gpu:4 each, 14-day max)
#   - GPU software ships as a Singularity container module, not a plain
#     python3/cuda module pair: `module load pytorch-gpu/2.8.0` only
#     prepends /cm/shared/containers/pytorch-gpu/2.8.0 to PATH, which
#     contains a `crun.pytorch-gpu` wrapper (symlink to singularity.sif)
#     -- you run commands INSIDE the container via
#     `crun.pytorch-gpu python3 ...`, plain `python3` still resolves to
#     the login node's system Python (no torch/numpy there).
#   - the container has torch+numpy+pandas; tqdm was missing and was
#     installed once via `crun.pytorch-gpu python3 -m pip install --user
#     tqdm` (writes to ~/.local, persists outside the container).
#
# Design: one array task per --order value, so the 9 orders run in
# parallel across nodes instead of sequentially in one job. Each task
# writes its own CSV (hyper_tnm_order<N>.csv); merge/plot afterward with
# plot_shatter_experiment.py --file_names hyper_tnm_order*.csv.
#
# Submit with:  sbatch submit_hpc.sbatch

#SBATCH --job-name=hyper_shatter
#SBATCH --account=<your-account>
#SBATCH --output=logs/shatter_%A_%a.out
#SBATCH --error=logs/shatter_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8                             # one task per entry in ORDERS below

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

ORDERS=(10 20 30 40 50 60 70 80 90)
ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}

crun.pytorch-gpu python3 run_shatter_experiment.py \
    --order "$ORDER" \
    --hidden_dim 4 16 64 256 \
    --num_layers 3 \
    --num_assign 50 \
    --train_steps 100000 \
    --lr 1e-4 \
    --seed 6 \
    --file_name "hyper_tnm_order${ORDER}.csv"
