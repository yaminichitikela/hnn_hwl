#!/bin/bash
# SLURM submission template for the FULL IMDB Q1 width sweep -- all 4
# datasets, one SLURM array task per dataset (parallel GPUs instead of
# sequential in one job), same array pattern as
# hypergraph_wl_wc/q3_shatter/submit_hpc.sbatch. Run this once the
# single-dataset trial (submit_hpc.sbatch, IMDB_wri_form) has confirmed
# the pipeline + tuning works end to end -- note task 2 below (wri_form)
# will re-run that same dataset; that's expected, not a mistake, and
# should reproduce the trial's numbers since tuning/splits are seeded.
#
# Requested walltime (48h) is sized for the largest dataset (IMDB_dir_genre,
# 3393 graphs, ~9x wri_form's 374) -- check `sacct -j <jobid>_<taskid>` per
# task once they finish and shorten for future resubmissions if they
# actually complete much faster.
#
# Submit with:  sbatch submit_full.sbatch

#SBATCH --job-name=imdb_q1_full
#SBATCH --account=<your-account>
#SBATCH --output=logs/q1_full_%A_%a.out
#SBATCH --error=logs/q1_full_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3                             # one task per entry in DATASETS below

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

DATASETS=(IMDB_dir_form IMDB_dir_genre IMDB_wri_form IMDB_wri_genre)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

crun.pytorch-gpu python3 q1_width_sweep.py --dataset "$DATASET"
