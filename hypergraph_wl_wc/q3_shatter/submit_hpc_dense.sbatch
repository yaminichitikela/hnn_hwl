#!/bin/bash
# SLURM submission for a DENSER width grid on the Q3 shattering sweep.
#
# Motivation: the paper currently tests only d in {4,16,64,256} (factor-4
# spacing between points). The observed family-size-shattered growth
# across those four points is not a clean linear-in-d signature (the
# 16->64 jump is far larger than the 4->16 or 64->256 jumps), and with
# only four points spaced this far apart, linear growth cannot be
# distinguished from sqrt(d), log(d), or grid-resolution noise -- exactly
# the limitation the paper's own Q3 discussion admits. This grid instead
# uses d in {4,8,16,32,64,128,256} (factor-2 spacing, 7 points, same
# endpoints as the original), dense enough to fit and compare candidate
# growth laws (linear vs sqrt(d) vs log(d)) via a proper log-log
# regression against the shattered family-size threshold per width.
#
# Writes to a SEPARATE file_name (hyper_tnm_dense_order<N>.csv) so the
# original hyper_tnm_order<N>.csv files -- already verified against the
# paper's Table 3 -- are never touched or at risk of being overwritten.
# The original 4 widths get re-run here too (for a self-consistent set of
# curves at every order), which is fine: this is a fresh, independent
# replication of those points, not a substitute for the original data.
#
# Same array-per-order pattern as submit_hpc.sbatch.
#
# Submit with:  sbatch submit_hpc_dense.sbatch

#SBATCH --job-name=hyper_shatter_dense
#SBATCH --account=<your-account>
#SBATCH --output=logs/shatter_dense_%A_%a.out
#SBATCH --error=logs/shatter_dense_%A_%a.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

ORDERS=(10 20 30 40 50 60 70 80 90)
ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}

crun.pytorch-gpu python3 run_shatter_experiment.py \
    --order "$ORDER" \
    --hidden_dim 4 8 16 32 64 128 256 \
    --num_layers 3 \
    --num_assign 50 \
    --train_steps 100000 \
    --lr 1e-4 \
    --seed 6 \
    --file_name "hyper_tnm_dense_order${ORDER}.csv"
