#!/bin/bash
# SLURM submission template for the IMDB Q2 depth sweep, first trial run.
# Same HPC conventions as imdb/q1/submit_hpc.sbatch (account=<your-account>,
# partition=gpu, crun.pytorch-gpu container wrapper).
#
# This is a single task (no array), scoped to one dataset via --dataset,
# for a first trial before submitting the full 4-dataset sweep. Output
# (including the printed tuned d_hidden/num_epochs and the VC/Rademacher
# bound table, now with the p_H/m column) goes to logs/q2_%j.out.
#
# Submit with:  sbatch submit_hpc.sbatch

#SBATCH --job-name=imdb_q2_wri_form
#SBATCH --account=<your-account>
#SBATCH --output=logs/q2_%j.out
#SBATCH --error=logs/q2_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

crun.pytorch-gpu python3 q2_depth_sweep.py --dataset IMDB_wri_form
