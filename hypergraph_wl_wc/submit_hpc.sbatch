#!/bin/bash
# SLURM submission for the Q1 width-sweep / Q2 depth-sweep experiments
# (4 datasets: imdb, rhg, steam, twitter -- x2 questions each = 8 runs).
# Configured for our institution's HPC cluster, same recipe as
# q3_shatter/submit_hpc.sbatch (verified interactively on a login node):
#   - account "<your-account>", partition "gpu"
#   - GPU software ships as a Singularity container module: `module load
#     pytorch-gpu/2.8.0` puts a `crun.pytorch-gpu` wrapper on PATH -- run
#     everything as `crun.pytorch-gpu python3 ...`, plain `python3` on the
#     login/compute node has no torch/numpy.
#
# These scripts read data straight off disk from
# data/dhg-bench/hgcls_data/<DATASET>/ via PROJECT_ROOT -- no GCS/env-var
# config needed; the Vertex AI cloud variants that once existed alongside
# these have been retired, this cluster is now the only place experiments
# run. QUICK_TEST has been flipped to False in all 8 scripts so this runs
# the real full sweep (auto-tuned depth/width/epochs), not the smoke test.
#
# Design: one array task per (dataset, question) pair, so all 8 run in
# parallel across nodes instead of sequentially. Each task's full stdout
# (all print()-based results/bound tables) lands in its own log file --
# these scripts don't write result files themselves.
#
# Submit with:  sbatch submit_hpc.sbatch   (run from hypergraph_wl_wc/)

#SBATCH --job-name=hnn_sweep
#SBATCH --account=<your-account>
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-7                             # one task per entry in TASKS below

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

TASKS=(
    "imdb/q1 q1_width_sweep.py"
    "imdb/q2 q2_depth_sweep.py"
    "rhg/q1 q1_width_sweep.py"
    "rhg/q2 q2_depth_sweep.py"
    "steam/q1 q1_width_sweep.py"
    "steam/q2 q2_depth_sweep.py"
    "twitter/q1 q1_width_sweep.py"
    "twitter/q2 q2_depth_sweep.py"
)
read -r SUBDIR SCRIPT <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

cd "$SUBDIR"
crun.pytorch-gpu python3 "$SCRIPT"
