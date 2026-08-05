#!/bin/bash
# SLURM submission template for the RHG Q1 width sweep (all 4 RHG datasets,
# run sequentially in one job -- q1_width_sweep.py loops over RHG_DATASETS
# itself, no --dataset flag needed here since there's no single-dataset
# trial to scope to first). Same HPC conventions as imdb/q1/submit_hpc.sbatch
# and hypergraph_wl_wc/q3_shatter/submit_hpc.sbatch (account=<your-account>, partition=gpu,
# crun.pytorch-gpu container wrapper).
#
# Requested walltime (3 days) is a rough guess since this is the first RHG
# run -- check actual time with `sacct -j <jobid>` once it finishes and
# adjust for future resubmissions.
#
# Submit with:  sbatch submit_hpc.sbatch

#SBATCH --job-name=rhg_q1
#SBATCH --account=<your-account>
#SBATCH --output=logs/q1_%j.out
#SBATCH --error=logs/q1_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

crun.pytorch-gpu python3 q1_width_sweep.py
