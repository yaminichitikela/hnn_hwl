#!/bin/bash
# SLURM submission template for the IMDB Q1 width sweep, first trial run.
# Configured for our institution's HPC cluster, same conventions verified working for
# hypergraph_wl_wc/q3_shatter (see that folder's submit_hpc.sbatch/README):
#   - account "<your-account>", partition "gpu"
#   - GPU software ships as a Singularity container module, not a plain
#     python3/cuda module pair: `module load pytorch-gpu/2.8.0` only
#     prepends /cm/shared/containers/pytorch-gpu/2.8.0 to PATH, which
#     contains a `crun.pytorch-gpu` wrapper -- run commands INSIDE the
#     container via `crun.pytorch-gpu python3 ...`; plain `python3` still
#     resolves to the login node's system Python (no torch/numpy there).
#
# This is a single task (no array), scoped to one dataset via --dataset,
# for a first trial before submitting the full 4-dataset sweep. Output
# (including the printed tuned num_layers/num_epochs and the VC/Rademacher
# bound table) goes to logs/q1_%j.out -- there is no separate results file
# yet, this script only reproduces what running q1_width_sweep.py locally
# prints to stdout.
#
# Submit with:  sbatch submit_hpc.sbatch

#SBATCH --job-name=imdb_q1_wri_form
#SBATCH --account=<your-account>
#SBATCH --output=logs/q1_%j.out
#SBATCH --error=logs/q1_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load pytorch-gpu/2.8.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

crun.pytorch-gpu python3 q1_width_sweep.py --dataset IMDB_wri_form
