#!/bin/bash
# SLURM submission for the Twitter-Friend VC-bound non-vacuousness demo
# (vc_bound_demo.py). Single dataset, no array needed. Much cheaper than
# the main width sweep (only d=2,3 instead of d=4,16,256,1024).
#
# Submit with:  sbatch submit_vc_bound_demo.sbatch

#SBATCH --job-name=twitter_vc_demo
#SBATCH --account=<your-account>
#SBATCH --output=logs/vc_demo_%j.out
#SBATCH --error=logs/vc_demo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

crun.pytorch-gpu python3 vc_bound_demo.py
