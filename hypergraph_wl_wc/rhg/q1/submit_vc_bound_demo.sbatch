#!/bin/bash
# SLURM submission for the RHG VC-bound non-vacuousness demo
# (vc_bound_demo.py) -- one array task per dataset, same pattern as
# imdb/q1/submit_vc_bound_demo.sbatch. Much cheaper than the main width
# sweep (only d=2,3 instead of d=4,16,256,1024), so a short walltime is
# enough.
#
# Submit with:  sbatch submit_vc_bound_demo.sbatch

#SBATCH --job-name=rhg_vc_demo
#SBATCH --account=<your-account>
#SBATCH --output=logs/vc_demo_%A_%a.out
#SBATCH --error=logs/vc_demo_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

DATASETS=(RHG_pyramid RHG_10 RHG_3 RHG_table)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

crun.pytorch-gpu python3 vc_bound_demo.py --dataset "$DATASET"
