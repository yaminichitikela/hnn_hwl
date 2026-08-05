"""
Q1 — width sweep, Twitter-Friend hypergraph dataset.

Question: how does the number of parameters (hidden width d) influence the
train-test generalisation gap? Hypergraph analogue of Morris et al.'s Q1
("WL meet VC", ICML 2023, arXiv:2301.11039, Sec 5).

Dataset: Twitter-Friend (twitter_friend.txt), 1310 hypergraphs, binary
labels. Per Feng, Han, Ying, Gao, "Hypergraph Isomorphism Computation"
(IEEE TPAMI 2024, arXiv:2307.14394 -- a different paper from
papers/rachmester.pdf, which is Carrasco et al.'s "Rademacher Meets
Colors", cited separately in q2_depth_sweep.py). Its Table 5 matches
directly: 1310 graphs, 2 classes, avg 21.6 vertices, avg 84.3 hyperedges.
"each hypergraph
is the friends of a specified user. The hyperedge is constructed by linking
the users who are friends. The label ... is to identify whether the user
posted the blog about 'national dog day' or 'respect tyler joseph'."

Protocol: same experimental shape as Morris et al.'s own code (wl_vs_vc/Q1/
main_weights.py): d in {4,16,256,1024}, batch_size=128, Adam lr=0.001,
90/10 split, CrossEntropyLoss, no dropout/weight decay. Unlike their
script, num_layers and num_epochs are NOT copied from their fixed L=5 /
500-epoch constants -- those were picked for "DD", a single much larger
GNN dataset, not for this smaller hypergraph dataset. Both are instead
auto-tuned by tune_hyperparams() below.

Architecture: HNN_slp per hnn-vc.pdf Eq. 1-2 (two-stage, single-layer
perceptrons throughout including readout).

VC bound: the/hnn_vc_full_proof.tex Theorem 1 (HNN analogue of Morris et al. 2023 Theorem 6), delta=1/p=2 (ReLU):
    VCdim(HNN_slp(d,L)) = O(L * P * log(2 * u * P))
Gap conversion: standard agnostic-PAC VC bound (Understanding Machine
Learning, Ch. 6, around Thm 6.7-6.8):
    gap <= sqrt(8*VCdim*ln(e*m/VCdim)/m) + sqrt(ln(2/delta)/(2m))

Run this yourself:
    cd "hypergraph_wl_wc/twitter/q1"
    python3 q1_width_sweep.py              # QUICK_TEST=False by default (full run)
    # set QUICK_TEST = True below first for a fast pipeline sanity-check --
    # num_layers/num_epochs are auto-tuned either way (see tune_hyperparams());
    # QUICK_TEST just caps how long that tuning search and the main sweep's
    # repeats/epochs are allowed to run.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../hyper graph algorithm research
TWITTER_DIR = PROJECT_ROOT / "data" / "dhg-bench" / "hgcls_data" / "TWITTER"
TWITTER_DATASETS = ["twitter_friend"]

# ---------------------------------------------------------------- Config --
QUICK_TEST = False   # True: tiny config to sanity-check the pipeline fast.
                     # Set False for the real Q1 sweep (full grid, tuned epochs).

TRAIN_FRAC = 0.90    # matches wl_vs_vc/Q1
LR         = 0.001   # matches wl_vs_vc/Q1
BATCH_SIZE = 128      # matches wl_vs_vc/Q1
DELTA      = 0.05    # confidence 1-delta = 95%

Q1_D_VALUES = [4, 16, 256, 1024]   # matches wl_vs_vc/Q1/main_weights.py hds
Q1_N_RUNS   = 10  # matches wl_vs_vc/Q1 num_reps; was 5; more runs -> tighter CI, no change to capacity/regularization

EVAL_EVERY = 10

# --- Hyperparameter auto-tuning -----------------------------------------
# num_layers and num_epochs used to be hardcoded to Morris et al.'s L=5 /
# 500 epochs. Those constants were picked for their "DD" dataset (a much
# larger, single-domain GNN benchmark) and were never re-derived for this
# smaller hypergraph dataset, so tune_hyperparams() below now picks both
# from a held-out slice the real Q1 sweep never trains or tests on.
TUNE_FRAC       = 0.15    # fraction of the dataset carved off for tuning only
TUNE_SEED       = 12345   # fixed seed for the carve-off (distinct from run_q1's seed=0..9)
TUNE_VAL_FRAC   = 0.80    # of the tuning slice: 80% train / 20% val, for early stopping
TUNE_D          = 16      # representative width used only to pick num_layers/num_epochs
MIN_LAYERS      = 2       # floor, in case HWL saturates trivially fast
MAX_TUNE_LAYERS = 10      # cap on how many HWL rounds we try before giving up on saturation
MAX_TUNE_EPOCHS = 500     # cap on how long the epoch-tuning run trains
TUNE_PATIENCE   = 30      # stop tuning run if val accuracy hasn't improved in this many epochs
TUNE_EVAL_EVERY = 5       # how often (in epochs) the tuning run checks val accuracy
TUNE_MIN_EPOCHS = 150     # floor on epochs before patience-based stopping is allowed to fire --
                          # without this, early validation checks can catch the model still at
                          # near-chance, noisy accuracy (before real learning has kicked in) and
                          # mistake that flat noise for convergence, stopping at best_epoch~0 and
                          # starving the whole real sweep of training (confirmed on steam/q1's
                          # identical pipeline: epochs=40, peaked at epoch 10 -- same signature
                          # twitter_friend showed here before this fix)
MIN_REAL_EPOCHS = 300     # hard floor on the epoch budget used for the REAL sweep (run_q1),
                          # regardless of what tuning found. tune_num_epochs()'s best_epoch is a
                          # single-point peak measured on a small validation slice, so even with
                          # TUNE_MIN_EPOCHS forcing the tuning run itself to keep going, a noisy
                          # early "peak" can still be the number that never gets beaten and end up
                          # governing the real sweep's budget. Until that noise is fixed at the
                          # source (e.g. a moving-average criterion), just floor the real sweep's
                          # budget directly instead of trusting the tuned number blindly.

if QUICK_TEST:
    print(">>> QUICK_TEST=True: tiny config, just to validate the pipeline runs end-to-end")
    EVAL_EVERY      = 5
    Q1_D_VALUES     = [4, 16]
    Q1_N_RUNS       = 1
    MAX_TUNE_LAYERS = 4
    MAX_TUNE_EPOCHS = 20
    TUNE_PATIENCE   = 5
    TUNE_EVAL_EVERY = 2
    TUNE_MIN_EPOCHS = 10
    MIN_REAL_EPOCHS = 20

DEVICE = (torch.device('mps') if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))
torch.manual_seed(42)


# ----------------------------------------------------------------- Data --
class Rec:
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
def compute_hwl_init(recs):
    """Round-0 (initial) colouring, feature-initialised (keyed on degree +
    sorted incident-cardinality multiset -- Twitter-Friend hyperedges vary
    in size, so this matters here). Returns the (all_nc, all_ec, sigs)
    state consumed by compute_hwl_round, so extra rounds can be added
    later without recomputing earlier rounds from scratch."""
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
    return all_nc, all_ec, sigs


def compute_hwl_round(recs, all_nc, all_ec, sigs):
    """One HWL refinement round (one hyperedge-colour update, then one
    vertex-colour update -- the same two-stage order an HNN layer
    performs), appending the new signature to sigs in place. Returns
    the updated (all_nc, all_ec, sigs) so another round can be chained
    on top without redoing this one."""
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

    return all_nc, all_ec, sigs


def compute_hwl(recs, L_max):
    """Runs L_max rounds via compute_hwl_round, from scratch. Kept as a
    thin convenience wrapper around compute_hwl_init/compute_hwl_round
    for callers that just want a fixed number of rounds in one shot."""
    all_nc, all_ec, sigs = compute_hwl_init(recs)
    for _ in range(L_max):
        all_nc, all_ec, sigs = compute_hwl_round(recs, all_nc, all_ec, sigs)
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
def find_saturating_layers(recs, max_layers, min_layers):
    """Smallest L at which HWL colour refinement saturates (one more round
    creates no new distinct colours) -- the natural, theory-grounded
    stopping point for this project's HWL/HNN correspondence, used instead
    of Morris et al.'s fixed L=5. Falls back to max_layers (with the
    caller warned via the returned `saturated=False`) if it never
    saturates within the cap."""
    all_nc, all_ec, sigs = compute_hwl_init(recs)
    prev_u = max(len(set(sigs[i][0])) for i in range(len(recs)))
    for L in range(1, max_layers + 1):
        all_nc, all_ec, sigs = compute_hwl_round(recs, all_nc, all_ec, sigs)
        u = max(len(set(sigs[i][L])) for i in range(len(recs)))
        if u == prev_u:
            return max(L - 1, min_layers), True
        prev_u = u
    return max_layers, False


def tune_num_epochs(tr_recs, val_recs, d, L, num_classes, max_epochs, patience, eval_every, min_epochs):
    """Trains at a representative width (d=TUNE_D) with early stopping on
    a held-out validation slice; no cosine schedule here (its T_max would
    require already knowing the final epoch count we're trying to find).
    Patience-based stopping is only allowed to fire once ep >= min_epochs,
    so a lucky/noisy near-chance validation reading in the first few checks
    can't be mistaken for convergence and cut training off before real
    learning has had a chance to start. Returns the epoch of best
    validation accuracy and that accuracy."""
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
    """Auto-tunes num_layers and num_epochs for this dataset instead of
    reusing Morris et al.'s fixed L=5 / 500-epoch constants (picked for
    "DD", a single much larger GNN dataset, not for this hypergraph
    dataset).

    Carves off TUNE_FRAC of the data purely for tuning (seeded separately
    from run_q1's own seed=0..9 splits), so the tuning validation slice
    never leaks into the folds the real Q1 sweep trains or tests on.
    Returns (num_layers, num_epochs, main_recs) where main_recs is the
    remaining pool run_q1 should use.
    """
    tune_recs, main_recs = make_split(recs, TUNE_FRAC, seed=TUNE_SEED)

    # Saturation is found on the FULL dataset (recs), not the 15% tune_recs
    # slice: unlike num_epochs tuning below, HWL colour refinement never
    # looks at labels or fits any parameters, so there is no leakage risk
    # in using the whole dataset here. Restricting this to tune_recs was an
    # artifact of reusing the same held-out slice for both searches, not a
    # deliberate choice -- and it's actively harmful, since a 15% sample can
    # miss whichever hypergraphs need the most refinement rounds, silently
    # undercounting u/p_H and the depth the model is actually given
    # (confirmed on RHG_10: the 15%-slice saturation check said L=6 was
    # enough, but checking against the full 85% main_recs pool afterward
    # showed colours still growing at L=7).
    layers, saturated = find_saturating_layers(recs, MAX_TUNE_LAYERS, MIN_LAYERS)
    if not saturated:
        print(f'  [{name}] WARNING: HWL colours still growing at L={MAX_TUNE_LAYERS} '
              f'(tuning cap) -- using {layers} layers anyway, saturation not confirmed')

    tt_train, tt_val = make_split(tune_recs, TUNE_VAL_FRAC, seed=TUNE_SEED)
    best_epoch, best_val_acc = tune_num_epochs(
        tt_train, tt_val, TUNE_D, layers, num_classes,
        MAX_TUNE_EPOCHS, TUNE_PATIENCE, TUNE_EVAL_EVERY, TUNE_MIN_EPOCHS)
    # margin so the real run isn't cut off exactly at the tuning run's best
    # snapshot -- give it one more patience window to keep improving.
    epochs = min(best_epoch + TUNE_PATIENCE, MAX_TUNE_EPOCHS)
    epochs = max(epochs, TUNE_EVAL_EVERY)
    # best_epoch is the single epoch with the highest measured validation
    # accuracy -- on this dataset's small tuning-validation slice, that
    # single-point measurement is noisy enough for an early lucky/unlucky
    # reading to "win" and never be beaten later even while the model keeps
    # improving underneath (confirmed on steam/q1's identical pipeline).
    # Rather than fix the noise in the tuning signal itself for now, just
    # floor the real sweep's epoch budget well above what that noisy
    # estimate alone would give.
    if epochs < MIN_REAL_EPOCHS:
        print(f'  [{name}] tuned epochs={epochs} looks too low to reflect real '
              f'convergence (tuning val set is small/noisy) -- flooring to '
              f'MIN_REAL_EPOCHS={MIN_REAL_EPOCHS} for the real sweep')
        epochs = MIN_REAL_EPOCHS

    print(f'  [{name}] tuned: num_layers={layers} (HWL saturation), '
          f'num_epochs={epochs} (val acc peaked {best_val_acc:.1f}% at epoch {best_epoch}, '
          f'd={TUNE_D}, on {len(tune_recs)}-graph tuning slice)')
    return layers, epochs, main_recs


# --------------------------------------------------------------- Q1 sweep -
def run_one(tr_recs, te_recs, d, L, num_classes, n_epochs, eval_every):
    """Adam lr=0.001, CrossEntropyLoss, no dropout/decay — matches
    wl_vs_vc/Q1/main_weights.py's optimiser/loss exactly; L and n_epochs
    are tuned values (see tune_hyperparams), not their fixed
    L=5/500-epoch constants."""
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


def run_q1(recs, num_classes, layers, n_epochs, eval_every):
    results = {}
    for d in Q1_D_VALUES:
        P = count_params(d, layers, num_classes)
        print(f'  d={d} L={layers} P={P:,}', flush=True)
        tr_all, te_all = [], []
        for run in range(Q1_N_RUNS):
            t0 = time.time()
            tr, te = make_split(recs, TRAIN_FRAC, seed=run)
            tr_a, te_a = run_one(tr, te, d, layers, num_classes, n_epochs, eval_every)
            tr_all.append(tr_a); te_all.append(te_a)
            g = tr_a[-1] - te_a[-1]
            print(f'    run {run+1}/{Q1_N_RUNS}: train={tr_a[-1]:.1f}% test={te_a[-1]:.1f}% '
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
        results[d] = {'train': np.array(tr_all), 'test': np.array(te_all), 'P': P}
    return results


# ------------------------------------------------------------------ Bound -
def vc_bound_pct(P, u, L, m, delta):
    vcdim = L * P * math.log(2 * u * P)
    if vcdim >= m:
        return vcdim, float('inf')
    bound = (math.sqrt(8 * vcdim * math.log(math.e * m / vcdim) / m)
             + math.sqrt(math.log(2 / delta) / (2 * m))) * 100
    return vcdim, bound


def rademacher_bound_pct(p_H, m, delta):
    """Carrasco et al., "Rademacher Meets Colors" -- see q2_depth_sweep.py
    for the full citation. Unlike vc_bound_pct, this does not depend on
    P, so it stays equally informative across every width d -- exactly
    where the VC bound above tends to go vacuous."""
    return (2 * math.sqrt(p_H / m) + 3 * math.sqrt(math.log(2 / delta) / (2 * m))) * 100


def check_q1_bound(name, recs, hwl_sigs, q1_results, layers):
    # same split make_split(recs, TRAIN_FRAC, seed=0) would produce (run_q1's
    # seed=0 run), computed directly to avoid building two throwaway Rec lists
    m = int(TRAIN_FRAC * len(recs))
    perm = np.random.default_rng(0).permutation(len(recs))
    train0_idx = list(perm[:m])
    # u used in the actual bound MUST be read at L=layers (matching the
    # model's depth) -- that is what the theorem requires. The extra
    # round (layers+1) is a validation check only: it must NEVER be fed
    # into vc_bound_pct, since HWL colours only ever grow or stay the
    # same round over round, so using the later round would silently
    # inflate u and loosen the bound rather than tighten it.
    u = max(len(set(hwl_sigs[i][layers])) for i in train0_idx)
    u_extra = max(len(set(hwl_sigs[i][layers + 1])) for i in train0_idx)
    sat = 'saturated' if u_extra == u else f'STILL GROWING ({u} -> {u_extra}); u used in bound may be an undercount'
    print(f'  HWL saturation check: u(L={layers})={u}, '
          f'u(L={layers+1})={u_extra} -- {sat}')

    # p_H: number of distinct HWL signatures among the m training
    # hypergraphs themselves (data-dependent), as opposed to u (the max
    # per-hypergraph colour count) -- same definition as Q2's p_H.
    # Neither u nor p_H depends on width d, so both are constant across
    # the whole Q1_D_VALUES sweep below.
    p_H = len({hwl_sigs[i][layers] for i in train0_idx})
    rb = rademacher_bound_pct(p_H, m, DELTA)

    print(f'\n--- {name}: Q1 VC + Rademacher bound check (m={m}, u={u}, p_H={p_H}) ---')
    print(f'{"d":>6} {"P":>10} {"VCdim":>12} {"VC bound":>10} {"Rad bound":>10} '
          f'{"emp gap":>9} {"VC ok?":>7} {"Rad ok?":>8}')
    rows = []
    for d in Q1_D_VALUES:
        P = q1_results[d]['P']
        vcdim, vb = vc_bound_pct(P, u, layers, m, DELTA)
        gap = float(q1_results[d]['train'][:, -1].mean() - q1_results[d]['test'][:, -1].mean())
        vbs = f'{vb:.1f}%' if vb < float('inf') else 'vacuous'
        vc_ok = 'YES' if gap <= vb else 'NO'
        rad_ok = 'YES' if gap <= rb else 'NO'
        print(f'{d:>6} {P:>10,} {vcdim:>12,.0f} {vbs:>10} {rb:>9.2f}% '
              f'{gap:>8.2f}% {vc_ok:>7} {rad_ok:>8}')
        rows.append({
            'dataset': name, 'm': m, 'u': u, 'p_H': p_H, 'layers': layers,
            'd': d, 'P': P, 'VCdim': vcdim,
            'VC_bound_pct': vb if vb < float('inf') else None,
            'Rad_bound_pct': rb, 'emp_gap_pct': gap,
            'VC_ok': vc_ok, 'Rad_ok': rad_ok,
        })
    return rows


# ---------------------------------------------------------------- Main ---
if __name__ == "__main__":
    print(f'Device: {DEVICE}  QUICK_TEST={QUICK_TEST}', flush=True)
    all_rows = []
    for name in TWITTER_DATASETS:
        path = TWITTER_DIR / f"{name}.txt"
        print(f'\n{"="*70}\n{name}  [Q1]\n{"="*70}')
        recs, labels, num_classes = load_dataset(path)
        print(f'{len(recs)} graphs, {num_classes} classes, feat_dim={FEAT_DIM}')

        layers, n_epochs, main_recs = tune_hyperparams(recs, num_classes, name)

        t0 = time.time()
        # Pass 1: standard L=layers rounds (matches the model's depth).
        all_nc, all_ec, hwl_sigs = compute_hwl_init(main_recs)
        for _ in range(layers):
            all_nc, all_ec, hwl_sigs = compute_hwl_round(main_recs, all_nc, all_ec, hwl_sigs)
        # Pass 2: one extra round, continuing from pass 1's state
        # (no recomputation of rounds 1..layers).
        all_nc, all_ec, hwl_sigs = compute_hwl_round(main_recs, all_nc, all_ec, hwl_sigs)
        print(f'HWL computed ({layers} rounds + 1 extra) in {time.time()-t0:.1f}s')

        q1_results = run_q1(main_recs, num_classes, layers, n_epochs, EVAL_EVERY)
        all_rows.extend(check_q1_bound(name, main_recs, hwl_sigs, q1_results, layers))

    import pandas as pd
    out_dir = Path('./results')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'q1_bound_check.csv'
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f'\nBound-check results saved to {csv_path}')
