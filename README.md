# Hypergraph Neural Network VC-Dimension and Rademacher Complexity Bounds

Code for the experiments accompanying the paper "Hypergraph Weisfeiler--Leman
Meets VC Dimension" (submitted to LoG 2026), which ties hypergraph neural
network (HNN) generalization to a hypergraph Weisfeiler--Leman (HWL) test,
via VC-dimension and Rademacher-complexity bounds.

## Layout

- `data/dhg-bench/hgcls_data/` -- the IMDB, RHG, Steam, and Twitter
  hypergraph classification datasets (Feng et al., "Hypergraph Isomorphism
  Computation") used for Q1 (width sweep) and Q2 (depth sweep).
- `hypergraph_wl_wc/imdb/`, `.../rhg/`, `.../steam/`, `.../twitter/` -- per-dataset
  Q1/Q2 sweep scripts (`q1_width_sweep.py`, `q2_depth_sweep.py`) and
  standalone VC-bound demo scripts (`vc_bound_demo.py`).
- `hypergraph_wl_wc/q3_shatter/` -- Q3, the shattering experiment: empirically
  lower-bounds `HNN_slp`'s true VC dimension against a synthetic hypergraph
  family. See its own `README.md` for usage.
- `hypergraph_wl_wc/q4_exact_bound/` -- Q4, the exact-bound experiment,
  checking the matching upper/lower VC-dimension bound against pairwise
  HWL-distinguishable hypergraph families. See its own `README.md`.

## Requirements

Each experiment folder that needs one has its own `requirements.txt`
(PyTorch-based; no other project-specific dependencies).

## Running locally

Each `q1_width_sweep.py` / `q2_depth_sweep.py` / `vc_bound_demo.py` script
is self-contained and can be run directly with `python3`; run with `--help`
for the available sweep parameters. `submit_hpc.sbatch` files are SLURM
templates for running the full sweeps on a GPU cluster -- edit the
`--account`/partition placeholders for your own cluster before submitting.
