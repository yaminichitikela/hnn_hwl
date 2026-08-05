"""
Q2 — depth sweep, IMDB hypergraph datasets.

Question: how does the number of HWL-distinguishable hypergraphs (colour
classes) influence the train-test generalisation gap as depth increases?
Hypergraph analogue of Morris et al.'s Q2 ("WL meet VC", ICML 2023,
arXiv:2301.11039, Sec 5) — same experimental design, same protocol,
different (hypergraph, not graph) input data/model.

Datasets: IMDB_dir_form, IMDB_dir_genre, IMDB_wri_form, IMDB_wri_genre — the
4 IMDB variants with a confirmed paper trail (HIC, Feng et al. 2024, IEEE
TPAMI; Zhang et al. 2025, ICML) and single-label targets. The two "-M"
files (IMDB_dir_genre_m, IMDB_wri_genre_m) are excluded: per HIC Sec 5.3.1,
"-M" denotes a genuinely multi-label task needing a different loss
(multi-label BCE) and metric (EMR / example-based accuracy) than the
single-label CrossEntropyLoss + argmax-accuracy pipeline here.

Protocol: same experimental shape as Morris et al.'s own code (wl_vs_vc/Q2/
main_colors.py — their public repo, cloned in this project): L in {0,...,6}
(the depth sweep -- this project's independent variable, same role as
Q1_D_VALUES in q1_width_sweep.py, so it is NOT auto-tuned), batch_size=128,
Adam lr=0.001, 90/10 split, CrossEntropyLoss, no dropout/weight decay.
Their own Q2 script only logs the *final* epoch's train/test accuracy per
run (no per-epoch curve); this version keeps a sparse per-epoch curve too
(EVAL_EVERY), which is strictly more information at negligible extra cost.
Unlike their script, d_hidden and num_epochs are NOT copied from their
fixed d=64 / 500-epoch constants -- those were picked for "DD", a single
much larger GNN dataset, not for these smaller IMDB hypergraphs. Both are
instead auto-tuned per dataset by tune_hyperparams() below (mirroring
q1_width_sweep.py's num_layers/num_epochs tuning, except width has no
structural saturation criterion the way HWL depth does, so it's picked via
a small validation-accuracy search over TUNE_WIDTH_CANDIDATES instead).

Architecture: HNN_slp per hnn-vc.pdf Eq. 1-2 (two-stage: edge update then
node update, single-layer perceptrons throughout, including the readout).
h_v^(0)=x_v is zero-padded (not learned) up to hidden width d, and the
multi-class linear readout generalises the theorem's scalar d+1-parameter
readout to a C-way one, C*(d+1) params. count_params() below computes the
resulting closed form P = 4Ld^2+2Ld+C(d+1) -- whose num_classes=1 case is
exactly the/hnn_vc_full_proof.tex's P_H=d(4dL+2L+1)+1 -- and asserts it
equals the trained model's real parameter count at construction time
(see HNNClassifier), rather than merely claiming consistency.

VC bound: the/hnn_vc_full_proof.tex Theorem 1, delta=1 / p=2 case (ReLU)
-- this project's own adaptation of Morris et al.'s (2023) Theorem 6 /
Prop. F.5 from GNNs to HNN_slp/HWL, not a formula stated in Morris et al.
themselves:
    VCdim(HNN_slp(d,L)) = O(L * P * log(2 * u * P))
where u = max number of distinct HWL colours in any single training
hypergraph at depth L (this project's own hypergraph analogue of Morris et
al.'s "color complexity" u for 1-WL on ordinary graphs — an extension this
project makes, not something proved in their paper for hypergraphs). Note
also that L=0 has no valid VC bound under this theorem (it's derived for
2L>=2 HWL floors); check_q2_bound() reports "n/a" rather than faking L=1.
Applying this scalar-output bound to a C-way argmax(CrossEntropyLoss)
classifier (via P's C-way readout term above) is a standard but NOT
rigorously reproven multiclass extension -- treat "VC ok?" below as
indicative, not a certified guarantee, whenever num_classes > 2.

Rademacher bound: p_H(L) = number of distinct HWL signatures in the
*training set* at depth L (see hnn_rhg.ipynb Sec 7-9 for the original,
RHG_pyramid-only version of this idea). The bound
    gap <= 2*sqrt(p_H/m) + 3*sqrt(ln(2/delta)/(2m))
is Carrasco, Deberaldini Netto, Martirosyan, Mehrab, Okoyomon, Graziani,
"Rademacher Meets Colors: More Expressivity, but at What Cost?" (NeurIPS
2025 Workshop, arXiv:2510.10101 -- ships as papers/rachmester.pdf), their
Lemma 2.2 + Proposition 2.3 (generalisation-from-Rademacher-complexity) and
Proposition 3.1 + Corollary 3.2 (R_S(F) <= sqrt(p/m) for colour-partitioned
classes with outputs in [-1,1]). An earlier draft of this project
mislabeled this citation as unverifiable -- it is real; "Carrasco" is the
paper's first author. The HWL/HNN_slp extension (replacing the paper's `p`
with `p_H`) is explicitly licensed by their own Sec 3.2 ("our analysis...
applies to any pair (A,T)... by replacing p with p_T"), and is worked out
step-by-step in the/hnn_rademacher_proof_v2.tex.

That derivation (like the VC one) requires HWL's initial colouring to be
*feature-initialised* -- see compute_hwl()'s docstring below -- for
"same HWL colour => same HNN output" to actually hold on datasets with
varying hyperedge cardinality (i.e. every IMDB file here).

check_q2_bound() additionally reports the ratio p_H/m directly, not just
the bound itself: since Rad bound = 2*sqrt(p_H/m) + ..., p_H/m is exactly
what the bound's growth across the depth sweep is driven by as HWL keeps
refining and p_H climbs toward m.

Gap conversion: standard agnostic-PAC VC uniform convergence bound
(Shalev-Shwartz & Ben-David, "Understanding Machine Learning", Ch. 6 "The
VC-Dimension", around Thm 6.7-6.8 — NOT Ch. 28 as an earlier draft of this
project mislabeled it; exact sub-theorem number not independently
re-verified page-by-page, double check before citing in the paper):
    gap <= sqrt(8*VCdim*ln(e*m/VCdim)/m) + sqrt(ln(2/delta)/(2m))

Run this yourself:
    cd "hypergraph_wl_wc/imdb/q2"
    python3 q2_depth_sweep.py              # QUICK_TEST=False by default (full run, all 4 datasets)
    python3 q2_depth_sweep.py --dataset IMDB_dir_genre   # just one dataset
    # set QUICK_TEST = True below first for a fast pipeline sanity-check --
    # d_hidden/num_epochs are auto-tuned either way (see tune_hyperparams());
    # QUICK_TEST just caps how long that tuning search and the main sweep's
    # repeats/epochs are allowed to run.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../hyper graph algorithm research
IMDB_DIR = PROJECT_ROOT / "data" / "dhg-bench" / "hgcls_data" / "IMDB"
IMDB_DATASETS = [
    "IMDB_dir_form",
    "IMDB_dir_genre",
    "IMDB_wri_form",
    "IMDB_wri_genre",
]

# ---------------------------------------------------------------- Config --
QUICK_TEST = False   # True: tiny config to sanity-check the pipeline fast.
                     # Set False for the real Q2 sweep (full grid, tuned epochs).

TRAIN_FRAC = 0.90    # matches wl_vs_vc/Q2
LR         = 0.001   # matches wl_vs_vc/Q2
BATCH_SIZE = 128      # matches wl_vs_vc/Q2
DELTA      = 0.05    # confidence 1-delta = 95%

Q2_L_VALUES = list(range(7))        # matches wl_vs_vc/Q2 num_layers = [0,...,6]
                                     # this is the depth SWEEP (independent variable) --
                                     # unlike d_hidden/num_epochs below, it is not tuned.
Q2_N_RUNS   = 10  # matches wl_vs_vc/Q2 num_reps; was 5; more runs -> tighter CI, no change to capacity/regularization

EVAL_EVERY = 10

# --- Hyperparameter auto-tuning -----------------------------------------
# d_hidden and num_epochs used to be hardcoded to Morris et al.'s d=64 /
# 500 epochs. Those constants were picked for their "DD" dataset (a much
# larger, single-domain GNN benchmark) and were never re-derived for these
# smaller/differently-shaped IMDB hypergraphs, so tune_hyperparams() below
# now picks both, per dataset, from a held-out slice the real Q2 sweep
# never trains or tests on. Width has no structural saturation criterion
# the way HWL depth does (q1_width_sweep.py's find_saturating_layers), so
# it's picked via a small validation-accuracy search instead.
TUNE_FRAC             = 0.15    # fraction of each dataset carved off for tuning only
TUNE_SEED             = 12345   # fixed seed for the carve-off (distinct from run_q2's seed=0..9)
TUNE_VAL_FRAC         = 0.80    # of the tuning slice: 80% train / 20% val, for early stopping
TUNE_L                = 3       # representative depth (middle of Q2_L_VALUES) used to tune width/epochs
TUNE_WIDTH_CANDIDATES = [16, 64, 256]  # candidate hidden widths to search; Morris et al.'s was a fixed 64
MAX_TUNE_EPOCHS = 500     # cap on how long each width candidate's tuning run trains
TUNE_PATIENCE   = 30      # stop a tuning run if val accuracy hasn't improved in this many epochs
TUNE_EVAL_EVERY = 5       # how often (in epochs) a tuning run checks val accuracy
TUNE_MIN_EPOCHS = 150     # floor on epochs before patience-based stopping is allowed to fire for any
                          # width candidate -- without this, an early validation check can catch the
                          # model still at near-chance, noisy accuracy (before real learning has
                          # kicked in) and mistake that flat noise for convergence, stopping at
                          # best_epoch~0 and starving the whole real sweep of training (confirmed on
                          # q1_width_sweep.py's identical pipeline, on steam_player: without this
                          # floor, real train accuracy was still climbing sharply at the resulting
                          # ~30-epoch cutoff, not plateaued)
MIN_REAL_EPOCHS = 300     # hard floor on the epoch budget used for the REAL sweep (run_q2),
                          # regardless of what tuning found. The winning width candidate's best_epoch
                          # is a single-point peak measured on a small validation slice, so even with
                          # TUNE_MIN_EPOCHS forcing each candidate's tuning run to keep going, a noisy
                          # early "peak" can still be the number that never gets beaten and end up
                          # governing the real sweep's budget. Until that noise is fixed at the
                          # source (e.g. a moving-average criterion), just floor the real sweep's
                          # budget directly instead of trusting the tuned number blindly.

if QUICK_TEST:
    print(">>> QUICK_TEST=True: tiny config, just to validate the pipeline runs end-to-end")
    EVAL_EVERY            = 5
    Q2_L_VALUES           = [0, 1, 2]
    Q2_N_RUNS             = 1
    TUNE_WIDTH_CANDIDATES = [16, 64]
    MAX_TUNE_EPOCHS       = 20
    TUNE_PATIENCE         = 5
    TUNE_EVAL_EVERY       = 2
    TUNE_MIN_EPOCHS       = 10
    MIN_REAL_EPOCHS       = 20

L_MAX = max(Q2_L_VALUES)

DEVICE = (torch.device('mps') if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))
torch.manual_seed(42)


# ----------------------------------------------------------------- Data --
class Rec:
    """One hypergraph. deg_v/card_e are the HWL initial colours
    (hwl-hnn.pdf Sec II-A: hyperdegree d_v, cardinality c_e — standard
    hypergraph definitions, independent of that paper's unrelated
    robustness-prediction claims).
    """
    __slots__ = ('members', 'incident', 'B', 'deg_v', 'card_e', 'label', 'n_v', 'n_e', 'x')

    def __init__(self, members, n_v, label):
        n_e = len(members)
        B = np.zeros((n_e, n_v), np.float32)
        incident = [[] for _ in range(n_v)]
        for ei, mems in enumerate(members):
            for v in mems:
                B[ei, v] = 1.0
                incident[v].append(ei)

        self.members = members
        self.incident = incident
        self.B = B
        self.deg_v = np.array([len(incident[v]) for v in range(n_v)], np.float32)
        self.card_e = np.array([len(mems) for mems in members], np.float32)
        self.label = label
        self.n_v = n_v
        self.n_e = n_e
        self.x = None


def load_dataset(path):
    with open(path) as f:
        raw_lines = [l.strip() for l in f.readlines()]

    all_recs, raw_labels = [], []
    idx = 0
    n_graphs = int(raw_lines[idx]); idx += 1
    for _ in range(n_graphs):
        parts = raw_lines[idx].split(); idx += 1
        n_v, n_e_hdr, label = int(parts[0]), int(parts[1]), int(parts[2])
        idx += 1  # skip placeholder feature line

        members = []
        for _ in range(n_e_hdr):
            nodes = list({int(v) for v in raw_lines[idx].split() if 0 <= int(v) < n_v})
            if nodes:
                members.append(nodes)
            idx += 1

        raw_labels.append(label)
        all_recs.append(Rec(members, n_v, label))

    assert idx == len(raw_lines), f"parser did not consume all lines: {idx} vs {len(raw_lines)}"

    # x_v = [norm_deg, norm_avg_card], dataset-wide min-max normalisation
    # (same construction as hnn_rhg.ipynb / server run/rhg_hwl_job.py).
    all_deg = np.concatenate([r.deg_v for r in all_recs])
    all_card = np.concatenate([r.card_e for r in all_recs])
    deg_min, deg_max = all_deg.min(), all_deg.max()
    card_min, card_max = all_card.min(), all_card.max()

    for r in all_recs:
        d_norm = (r.deg_v - deg_min) / max(deg_max - deg_min, 1e-8)
        c_norm = (r.card_e - card_min) / max(card_max - card_min, 1e-8)
        avg_card = np.array([
            c_norm[r.incident[v]].mean() if r.incident[v] else 0.0
            for v in range(r.n_v)
        ], np.float32)
        r.x = np.stack([d_norm, avg_card], axis=1)

    # remap labels to a contiguous 0..K-1 range for CrossEntropyLoss
    label_set = sorted(set(raw_labels))
    num_classes = len(label_set)
    if label_set != list(range(num_classes)):
        remap = {lab: i for i, lab in enumerate(label_set)}
        for r in all_recs:
            r.label = remap[r.label]
        raw_labels = [remap[l] for l in raw_labels]

    return all_recs, raw_labels, num_classes


FEAT_DIM = 2  # x_v = [norm_deg, norm_avg_card]


# ------------------------------------------------------------------ HWL --
def compute_hwl(recs, L_max):
    """Two-stage HWL colour refinement (hwl-hnn.pdf Sec II-A definitions;
    the two-stage update *shape* also matches hnn-vc.pdf Eq. 1's edge-then-
    node structure). Computed once up to L_max in a single pass; every
    depth L in Q2_L_VALUES just indexes into this precomputed list.

    Initial node colour is *feature-initialised*: keyed on
    (degree, sorted multiset of incident hyperedge cardinalities), not
    degree alone. This is required for the "same HWL colour => same HNN
    output" equivalence (the/hnn_vc_full_proof.tex's Lemma; also underlies
    the Rademacher bound above) to actually hold: the network's real input
    is x_v = [norm_deg, norm_avg_card], and degree-only colouring
    (hwl-hnn.pdf's weaker Lemma 2) only guarantees x_v is determined by the
    colour when cardinality is constant across the dataset (e.g.
    RHG_pyramid's 3-uniform hyperedges). On any dataset with varying
    cardinality -- every IMDB file here -- two same-degree nodes can have
    different avg_card, so degree-only colouring would wrongly group them.
    Keying on the full incident-cardinality multiset fixes this: it
    determines avg_card exactly, on every dataset.
    Initial edge colour = integer cardinality (unchanged; safe as-is).
    """
    init_nc_map = {}
    all_nc = []
    for r in recs:
        nc = []
        for v in range(r.n_v):
            deg = len(r.incident[v])
            card_multiset = tuple(sorted(int(r.card_e[e]) for e in r.incident[v]))
            key = (deg, card_multiset)
            if key not in init_nc_map:
                init_nc_map[key] = len(init_nc_map)
            nc.append(init_nc_map[key])
        all_nc.append(nc)

    all_ec = [[len(mems) for mems in r.members] for r in recs]
    sigs = [[tuple(sorted(all_nc[i]))] for i in range(len(recs))]

    for _ in range(L_max):
        new_ec_map = {}
        new_all_ec = []
        for i, r in enumerate(recs):
            new_ec = []
            for ei, mems in enumerate(r.members):
                k = (all_ec[i][ei], tuple(sorted(all_nc[i][v] for v in mems)))
                if k not in new_ec_map:
                    new_ec_map[k] = len(new_ec_map)
                new_ec.append(new_ec_map[k])
            new_all_ec.append(new_ec)
        all_ec = new_all_ec

        new_nc_map = {}
        new_all_nc = []
        for i, r in enumerate(recs):
            new_nc = []
            for v in range(r.n_v):
                k = (all_nc[i][v], tuple(sorted(all_ec[i][e] for e in r.incident[v])))
                if k not in new_nc_map:
                    new_nc_map[k] = len(new_nc_map)
                new_nc.append(new_nc_map[k])
            new_all_nc.append(new_nc)
        all_nc = new_all_nc

        for i in range(len(recs)):
            sigs[i].append(tuple(sorted(all_nc[i])))

    return sigs


# ---------------------------------------------------------- HNN_slp model -
class HNNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d, bias=False)
        self.W3 = nn.Linear(d, d)
        self.W4 = nn.Linear(d, d, bias=False)

    def forward(self, h_v, h_e, B, node_mask, he_mask):
        Bh_v = torch.bmm(B, h_v)
        h_e = torch.relu(self.W1(h_e) + self.W2(Bh_v)) * he_mask.unsqueeze(-1)
        BTh_e = torch.bmm(B.transpose(1, 2), h_e)
        h_v = torch.relu(self.W3(h_v) + self.W4(BTh_e)) * node_mask.unsqueeze(-1)
        return h_v, h_e


class HNNClassifier(nn.Module):
    """h_v^(0) = x_v exactly, per hnn-vc.pdf Eq. 1's initial condition:
    raw features are zero-padded (not learned) up to hidden width d, so
    there is no input-embedding parameter outside the theorem's P_H. The
    multi-class linear readout generalises the theorem's scalar readout
    (d+1 params) to a C-way one (C*(d+1) params); param_count_formula()
    below is the resulting closed form, which count_params() is asserted
    to match exactly -- see the/hnn_vc_full_proof.tex for the num_classes=1
    case this generalises."""
    def __init__(self, feat_dim, d, n_layers, num_classes):
        super().__init__()
        assert feat_dim <= d, f"feat_dim={feat_dim} must be <= hidden width d={d} to zero-pad"
        self.d = d
        self.feat_dim = feat_dim
        self.layers = nn.ModuleList([HNNLayer(d) for _ in range(n_layers)])
        self.readout = nn.Linear(d, num_classes)

    def forward(self, x, B, node_mask, he_mask):
        pad = x.new_zeros(*x.shape[:-1], self.d - self.feat_dim)
        h_v = torch.cat([x, pad], dim=-1) * node_mask.unsqueeze(-1)
        h_e = torch.zeros(B.shape[0], B.shape[1], self.d, device=x.device)
        for layer in self.layers:
            h_v, h_e = layer(h_v, h_e, B, node_mask, he_mask)
        h_G = (h_v * node_mask.unsqueeze(-1)).sum(dim=1)
        return self.readout(h_G)


def param_count_formula(d, n_layers, num_classes):
    """Closed-form parameter count for HNNClassifier: the/hnn_vc_full_proof.tex's
    P_H = d(4dL+2L+1)+1 generalised from a scalar readout (num_classes=1,
    d+1 params) to a C-way linear readout (C*(d+1) params); the zero-padded
    input has no trainable parameters, so every other term is unchanged."""
    return 4 * n_layers * d**2 + 2 * n_layers * d + num_classes * (d + 1)


def count_params(d, n_layers, num_classes):
    n = sum(p.numel() for p in HNNClassifier(FEAT_DIM, d, n_layers, num_classes).parameters())
    formula = param_count_formula(d, n_layers, num_classes)
    assert n == formula, f"actual param count {n} != closed-form P_H {formula}"
    return n


# ------------------------------------------------------ Training utilities
class GraphDS(Dataset):
    def __init__(self, recs): self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return r.x, r.B, int(r.label)


def collate(batch):
    max_nv = max(x.shape[0] for x, _, _ in batch)
    max_ne = max(B.shape[0] for _, B, _ in batch)
    Bs = len(batch); fd = batch[0][0].shape[1]
    xb = np.zeros((Bs, max_nv, fd), np.float32)
    Bb = np.zeros((Bs, max_ne, max_nv), np.float32)
    nm = np.zeros((Bs, max_nv), np.float32)
    hem = np.zeros((Bs, max_ne), np.float32)
    yb = np.empty(Bs, np.int64)
    for i, (x, B, y) in enumerate(batch):
        nv, ne = x.shape[0], B.shape[0]
        xb[i, :nv] = x
        Bb[i, :ne, :nv] = B
        nm[i, :nv] = 1.0
        hem[i, :ne] = 1.0
        yb[i] = y
    return (torch.from_numpy(xb), torch.from_numpy(Bb),
            torch.from_numpy(nm), torch.from_numpy(hem),
            torch.from_numpy(yb))


def make_split(recs, frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(recs))
    k = int(frac * len(recs))
    return [recs[i] for i in perm[:k]], [recs[i] for i in perm[k:]]


def acc(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, B, nm, hem, y in loader:
            x, B, nm, hem, y = (t.to(DEVICE) for t in (x, B, nm, hem, y))
            pred = model(x, B, nm, hem).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total * 100


# ------------------------------------------------------ Hyperparameter tuning
def tune_num_epochs(tr_recs, val_recs, d, L, num_classes, max_epochs, patience, eval_every, min_epochs):
    """Trains one (d, L) config with early stopping on a held-out
    validation slice; no cosine schedule here (its T_max would require
    already knowing the final epoch count we're trying to find). Returns
    the epoch of best validation accuracy and that accuracy -- used both
    to score width candidates and to derive the final num_epochs.
    Patience-based stopping is only allowed to fire once ep >= min_epochs,
    so a lucky/noisy near-chance validation reading in the first few checks
    can't be mistaken for convergence and cut training off before real
    learning has had a chance to start."""
    model = HNNClassifier(FEAT_DIM, d, L, num_classes).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    trl = DataLoader(GraphDS(tr_recs), BATCH_SIZE, shuffle=True, collate_fn=collate)
    vall = DataLoader(GraphDS(val_recs), BATCH_SIZE * 2, shuffle=False, collate_fn=collate)
    best_val, best_epoch, since_improve = -1.0, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for x, B, nm, hem, y in trl:
            x, B, nm, hem, y = (t.to(DEVICE) for t in (x, B, nm, hem, y))
            opt.zero_grad()
            crit(model(x, B, nm, hem), y).backward()
            opt.step()
        if ep % eval_every == 0:
            va = acc(model, vall)
            if va > best_val + 1e-6:
                best_val, best_epoch, since_improve = va, ep, 0
            else:
                since_improve += eval_every
            if ep >= min_epochs and since_improve >= patience:
                break
    return best_epoch, best_val


def tune_hyperparams(recs, num_classes, name):
    """Auto-tunes d_hidden and num_epochs for this dataset instead of
    reusing Morris et al.'s fixed d=64 / 500-epoch constants (picked for
    "DD", a single much larger GNN dataset, not for these hypergraphs).

    Carves off TUNE_FRAC of the data purely for tuning (seeded separately
    from run_q2's own seed=0..9 splits), so the tuning validation slice
    never leaks into the folds the real Q2 sweep trains or tests on.

    Unlike depth (q1_width_sweep.py's HWL-saturation criterion), width has
    no structural stopping point, so each candidate in TUNE_WIDTH_CANDIDATES
    is trained (at a fixed representative depth TUNE_L) with early stopping,
    and the candidate with the best validation accuracy is kept -- its own
    best-epoch point (plus a patience margin) then becomes num_epochs, so
    a single set of tuning runs picks both values together.

    Returns (d_hidden, num_epochs, main_recs) where main_recs is the
    remaining pool run_q2 should use.
    """
    tune_recs, main_recs = make_split(recs, TUNE_FRAC, seed=TUNE_SEED)
    tt_train, tt_val = make_split(tune_recs, TUNE_VAL_FRAC, seed=TUNE_SEED)

    best_d, best_epoch, best_val_acc = TUNE_WIDTH_CANDIDATES[0], 0, -1.0
    for d in TUNE_WIDTH_CANDIDATES:
        ep, va = tune_num_epochs(tt_train, tt_val, d, TUNE_L, num_classes,
                                  MAX_TUNE_EPOCHS, TUNE_PATIENCE, TUNE_EVAL_EVERY, TUNE_MIN_EPOCHS)
        print(f'  [{name}] width candidate d={d} (L={TUNE_L}): '
              f'val acc peaked {va:.1f}% at epoch {ep}')
        if va > best_val_acc + 1e-6:   # strict improvement only -> ties keep the smaller, earlier-tried width
            best_d, best_epoch, best_val_acc = d, ep, va

    # margin so the real run isn't cut off exactly at the tuning run's best
    # snapshot -- give it one more patience window to keep improving.
    epochs = min(best_epoch + TUNE_PATIENCE, MAX_TUNE_EPOCHS)
    epochs = max(epochs, TUNE_EVAL_EVERY)
    # best_epoch is a single-point peak measured on a small validation
    # slice, so it can still be an unreliable underestimate even with the
    # min_epochs warm-up above (confirmed on q1_width_sweep.py's identical
    # pipeline). Floor the real sweep's budget directly instead of
    # trusting it blindly.
    if epochs < MIN_REAL_EPOCHS:
        print(f'  [{name}] tuned epochs={epochs} looks too low to reflect real '
              f'convergence (tuning val set is small/noisy) -- flooring to '
              f'MIN_REAL_EPOCHS={MIN_REAL_EPOCHS} for the real sweep')
        epochs = MIN_REAL_EPOCHS

    print(f'  [{name}] tuned: d_hidden={best_d} (best val acc {best_val_acc:.1f}% '
          f'among {TUNE_WIDTH_CANDIDATES} at L={TUNE_L}), num_epochs={epochs}')
    return best_d, epochs, main_recs


# --------------------------------------------------------------- Q2 sweep -
def run_one(tr_recs, te_recs, d, L, num_classes, n_epochs, eval_every):
    """Adam lr=0.001, CrossEntropyLoss, no dropout/decay — matches
    wl_vs_vc/Q2/main_colors.py's optimiser/loss exactly; d and n_epochs
    are per-dataset tuned values (see tune_hyperparams), not their fixed
    d=64/500-epoch constants."""
    model = HNNClassifier(FEAT_DIM, d, L, num_classes).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    # Pre-scheduled (epoch-count-only, not validation-dependent) cosine decay:
    # smooths late-training oscillation without acting as an implicit
    # regularizer the way validation-based early stopping would (Goodfellow
    # et al., "Deep Learning", Sec. 7.8), so the VC/Rademacher bound-vs-gap
    # comparison against the plain HNN_slp function class stays valid.
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    trl = DataLoader(GraphDS(tr_recs), BATCH_SIZE, shuffle=True, collate_fn=collate)
    tel = DataLoader(GraphDS(te_recs), BATCH_SIZE * 2, shuffle=False, collate_fn=collate)
    tr_a, te_a = [], []
    for ep in range(1, n_epochs + 1):
        model.train()
        for x, B, nm, hem, y in trl:
            x, B, nm, hem, y = (t.to(DEVICE) for t in (x, B, nm, hem, y))
            opt.zero_grad()
            crit(model(x, B, nm, hem), y).backward()
            opt.step()
        sched.step()
        if ep % eval_every == 0:
            tr_a.append(acc(model, trl))
            te_a.append(acc(model, tel))
    return np.array(tr_a), np.array(te_a)


def run_q2(recs, num_classes, d_hidden, n_epochs, eval_every):
    results = {}
    for L in Q2_L_VALUES:
        P = count_params(d_hidden, L, num_classes)
        print(f'  L={L} d={d_hidden} P={P:,}', flush=True)
        tr_all, te_all = [], []
        for run in range(Q2_N_RUNS):
            t0 = time.time()
            tr, te = make_split(recs, TRAIN_FRAC, seed=run)
            tr_a, te_a = run_one(tr, te, d_hidden, L, num_classes, n_epochs, eval_every)
            tr_all.append(tr_a); te_all.append(te_a)
            g = tr_a[-1] - te_a[-1]
            print(f'    run {run+1}/{Q2_N_RUNS}: train={tr_a[-1]:.1f}% test={te_a[-1]:.1f}% '
                  f'gap={g:.1f}% ({time.time()-t0:.1f}s)', flush=True)
            # Convergence check: last few evaluated points of the train-accuracy
            # trajectory (one point per `eval_every` epochs), so we can tell
            # whether training had already flattened out by the last epoch or
            # was still climbing when the epoch budget cut it off.
            tail_tr = ', '.join(f'{x:.1f}' for x in tr_a[-5:])
            tail_te = ', '.join(f'{x:.1f}' for x in te_a[-5:])
            print(f'      trajectory (last {min(5, len(tr_a))} of {len(tr_a)} evals, '
                  f'every {eval_every} epochs): train=[{tail_tr}] test=[{tail_te}]',
                  flush=True)
        results[L] = {'train': np.array(tr_all), 'test': np.array(te_all), 'P': P}
    return results


# ------------------------------------------------------------------ Bound -
def vc_bound_pct(P, u, L, m, delta):
    """the/hnn_vc_full_proof.tex Theorem 1 (HNN analogue of Morris et al. 2023 Theorem 6) (delta=1, p=2 / ReLU case):
    VCdim <= L*P*log(2*u*P). Converted to a generalisation-gap percentage
    via the standard agnostic-PAC VC uniform convergence bound (UML Ch.6,
    around Thm 6.7-6.8 -- verify the exact sub-number before citing)."""
    vcdim = L * P * math.log(2 * u * P)
    if vcdim >= m:
        return vcdim, float('inf')
    bound = (math.sqrt(8 * vcdim * math.log(math.e * m / vcdim) / m)
             + math.sqrt(math.log(2 / delta) / (2 * m))) * 100
    return vcdim, bound


def rademacher_bound_pct(p_H, m, delta):
    """Carrasco et al., "Rademacher Meets Colors" (NeurIPS 2025 Workshop,
    arXiv:2510.10101), Lemma 2.2 + Prop 2.3 + Prop 3.1 + Corollary 3.2,
    extended from 1-WL/GNN to HWL/HNN_slp via their Sec 3.2 (licensed:
    "applies to any pair (A,T)... by replacing p with p_T"; worked out in
    the/hnn_rademacher_proof_v2.tex):
        gap <= 2*sqrt(p_H/m) + 3*sqrt(ln(2/delta)/(2m))
    Requires feature-initialised HWL colouring (see compute_hwl) for the
    "same colour => same output" premise to hold. Unlike vc_bound_pct,
    this does not depend on P, so it stays informative at every depth L --
    exactly where the VC bound above tends to go vacuous."""
    return (2 * math.sqrt(p_H / m) + 3 * math.sqrt(math.log(2 / delta) / (2 * m))) * 100


def check_q2_bound(name, recs, hwl_sigs, q2_results, d_hidden):
    # same split make_split(recs, TRAIN_FRAC, seed=0) would produce (run_q2's
    # seed=0 run), computed directly to avoid building two throwaway Rec lists
    m = int(TRAIN_FRAC * len(recs))
    perm = np.random.default_rng(0).permutation(len(recs))
    train0_idx = list(perm[:m])

    print(f'\n--- {name}: Q2 VC + Rademacher bound check (m={m}, d={d_hidden}) ---')
    print(f'{"L":>4} {"P":>10} {"u":>6} {"p_H":>6} {"p_H/m":>8} {"VC bound":>10} '
          f'{"Rad bound":>10} {"emp gap":>9} {"VC ok?":>7} {"Rad ok?":>8}')
    rows = []
    for L in Q2_L_VALUES:
        u_L = max(len(set(hwl_sigs[i][L])) for i in train0_idx)
        p_H = len({hwl_sigs[i][L] for i in train0_idx})
        p_H_over_m = p_H / m
        P = q2_results[L]['P']
        rb = rademacher_bound_pct(p_H, m, DELTA)
        gap = float(q2_results[L]['train'][:, -1].mean() - q2_results[L]['test'][:, -1].mean())
        if L == 0:
            # Theorem (the/hnn_vc_full_proof.tex) is derived for 2L>=2
            # hidden floors; L=0 (no message passing) is out of its scope,
            # so leave the VC bound undefined instead of faking L=1.
            vbs, vc_ok, vb_val = 'n/a (L=0)', 'n/a', None
        else:
            _, vb = vc_bound_pct(P, u_L, L, m, DELTA)
            vbs = f'{vb:.1f}%' if vb < float('inf') else 'vacuous'
            vc_ok = 'YES' if gap <= vb else 'NO'
            vb_val = vb if vb < float('inf') else None
        rad_ok = 'YES' if gap <= rb else 'NO'
        print(f'{L:>4} {P:>10,} {u_L:>6} {p_H:>6} {p_H_over_m:>8.4f} {vbs:>10} {rb:>9.2f}% '
              f'{gap:>8.2f}% {vc_ok:>7} {rad_ok:>8}')
        rows.append({
            'dataset': name, 'm': m, 'd': d_hidden, 'L': L, 'P': P,
            'u': u_L, 'p_H': p_H, 'p_H_over_m': p_H_over_m,
            'VC_bound_pct': vb_val, 'Rad_bound_pct': rb, 'emp_gap_pct': gap,
            'VC_ok': vc_ok, 'Rad_ok': rad_ok,
        })
    return rows


# ---------------------------------------------------------------- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=IMDB_DATASETS, default=None,
                         help="run only this one dataset instead of all 4 "
                              "(e.g. for a first cluster trial run)")
    args = parser.parse_args()
    datasets_to_run = [args.dataset] if args.dataset else IMDB_DATASETS

    print(f'Device: {DEVICE}  QUICK_TEST={QUICK_TEST}', flush=True)
    all_rows = []
    for name in datasets_to_run:
        path = IMDB_DIR / f"{name}.txt"
        print(f'\n{"="*70}\n{name}  [Q2]\n{"="*70}')
        recs, labels, num_classes = load_dataset(path)
        print(f'{len(recs)} graphs, {num_classes} classes, feat_dim={FEAT_DIM}')

        d_hidden, n_epochs, main_recs = tune_hyperparams(recs, num_classes, name)

        t0 = time.time()
        hwl_sigs = compute_hwl(main_recs, L_MAX)
        print(f'HWL computed (L_max={L_MAX}) in {time.time()-t0:.1f}s')

        q2_results = run_q2(main_recs, num_classes, d_hidden, n_epochs, EVAL_EVERY)
        all_rows.extend(check_q2_bound(name, main_recs, hwl_sigs, q2_results, d_hidden))

    import pandas as pd
    out_dir = Path('./results')
    out_dir.mkdir(exist_ok=True)
    suffix = f'_{args.dataset}' if args.dataset else ''
    csv_path = out_dir / f'q2_bound_check{suffix}.csv'
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f'\nBound-check results saved to {csv_path}')
