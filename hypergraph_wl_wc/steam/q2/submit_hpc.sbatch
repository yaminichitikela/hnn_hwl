#!/bin/bash
# SLURM submission template for the Steam-Player Q2 depth sweep (single
# dataset, stream_player). Same HPC conventions as imdb/q1/submit_hpc.sbatch.
#
# Requested walltime (24h) is a rough guess for a 2048-graph single-dataset
# full run -- check actual time with `sacct -j <jobid>` once it finishes.
#
# Submit with:  sbatch submit_hpc.sbatch

#SBATCH --job-name=steam_q2
#SBATCH --account=<your-account>
#SBATCH --output=logs/q2_%j.out
#SBATCH --error=logs/q2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

crun.pytorch-gpu python3 q2_depth_sweep.py
