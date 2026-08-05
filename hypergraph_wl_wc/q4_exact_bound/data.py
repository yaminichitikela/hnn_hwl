"""
Hypergraph family construction for the prop:upper-nonuniform exact-bound
test ("Q4"): VCdim_{H_{d,n}^bool}(HNN_mlp(L)) = m^H_{n,d,L}, where
m^H_{n,d,L} is the largest number of hypergraphs of order at most n with
d-dimensional boolean features that are pairwise HWL-distinguishable
after L rounds.

Unlike Q1/Q2/Q3, which all load or construct ONE fixed family and sweep
width/depth over it, this experiment needs to CONSTRUCT the family the
theorem is about: a maximal set of pairwise-HWL-distinguishable
hypergraphs (found by sampling a large random pool and grouping by stable
HWL colour signature), plus one deliberately colour-clashing hypergraph
added on top -- the two directions the paper's proof actually needs:

  - `base_indices`: one representative per distinct signature found in the
    pool. This is a LOWER BOUND on the true m^H_{n,d,L} (the pool might
    not contain every achievable signature, so the true maximum could be
    larger; it is not an upper bound estimate). Section run_exact_bound_
    experiment.py tests whether a sufficiently wide HNN_mlp(L) can
    SHATTER this exact family -- the lower-bound (construction) half of
    the theorem.

  - twin: one further hypergraph sharing its signature with an existing
    base-family member. Adding it to the base family gives a set of size
    k+1 that pigeonhole guarantees is NOT pairwise HWL-distinguishable
    (two members share a colour). The upper-bound half of the theorem
    says no HNN_mlp, at any width, can shatter this augmented family --
    specifically, it can never separate the twin pair, so any of the 50
    random label draws that assign them different target labels is
    provably unfittable.

    NOTE on how the twin is obtained: an early calibration run showed
    that random hypergraphs essentially never collide in HWL signature at
    pool sizes small enough for a tractable base family (order<=8,
    pool<=80 never produced a single natural collision; it took a pool of
    ~1000 to see a ~6.6% collision rate, which would also make the base
    family itself ~1000 hypergraphs -- likely too large to reliably
    shatter within a modest width sweep, and no longer a small,
    Q3-comparable family). Rather than gamble on a big enough pool to hit
    a lucky natural collision, `make_twin` below constructs a twin
    DIRECTLY: an isomorphic copy of a chosen base-family member, with
    vertex labels permuted. HWL colour refinement is provably
    isomorphism-invariant (every step here uses only structural/incidence
    information and order-independent sorted multisets, never absolute
    vertex indices), so the copy is guaranteed -- not just likely -- to
    share its original's exact stable signature, for any n, d, or pool
    size. `build_families` below still also detects and reports any
    natural collisions the pool happens to contain, as a free diagnostic,
    but the augmented family's twin is never dependent on finding one.

HWL here is the feature-initialised variant from the paper's Preliminaries
(sec:hwl): c_v^(0)(v) is the equivalence class of x_v (for boolean
features, this is just the exact boolean vector, a finite alphabet of at
most 2^d values), and c_e^(0)(e) := (|e|, [x_e]) pairs the hyperedge's
cardinality with the equivalence class of its own boolean feature x_e.
Colour maps are GLOBAL across the whole pool (not per-hypergraph), so
that identical colours occurring in different hypergraphs are correctly
recognised as identical when comparing signatures across the pool --
this is the one place this file's HWL implementation must differ from
q1_width_sweep.py's (there, only within-dataset consistency is needed
since bound-checking never compares colours *across* hypergraphs).

The stable colouring of any hypergraph of order at most n is reached
within n refinement rounds (used throughout the paper's own proofs, e.g.
lem:hnn-simulates-hwl), so this file always runs exactly L=n rounds.
"""
import numpy as np


def random_hypergraph(n_v, d, rng, min_edges=None, max_edges=None,
                       min_card=2, max_card=None):
    """One random hypergraph of exactly n_v vertices, with random
    hyperedges of varying cardinality and random d-dimensional boolean
    features on every vertex and every hyperedge."""
    max_card = min(max_card or n_v, n_v)
    min_edges = min_edges if min_edges is not None else max(1, n_v // 3)
    max_edges = max_edges if max_edges is not None else 2 * n_v
    n_e = int(rng.integers(min_edges, max_edges + 1))

    members = []
    for _ in range(n_e):
        card = int(rng.integers(min_card, max_card + 1))
        verts = sorted(rng.choice(n_v, size=card, replace=False).tolist())
        members.append(verts)

    incident = [[] for _ in range(n_v)]
    for ei, mem in enumerate(members):
        for v in mem:
            incident[v].append(ei)

    x_v = rng.integers(0, 2, size=(n_v, d)).astype(np.float32)
    x_e = rng.integers(0, 2, size=(n_e, d)).astype(np.float32)

    return {'members': members, 'incident': incident, 'n_v': n_v, 'n_e': n_e,
            'x_v': x_v, 'x_e': x_e}


def make_twin(hg, rng):
    """An isomorphic copy of hg with vertex labels randomly permuted.
    HWL colour refinement is isomorphism-invariant (every colour update
    in this file uses only structural/incidence information and
    order-independent sorted multisets, never absolute vertex indices),
    so this copy is GUARANTEED to reach the exact same stable signature
    as hg -- a distinct hypergraph object, but not pairwise
    HWL-distinguishable from it, regardless of n, d, or how it was
    generated. This is what gives the augmented (pigeonhole) family its
    twin, without depending on a random pool happening to contain a
    natural collision (see the module docstring for why that turned out
    to be unreliable at tractable pool sizes)."""
    n_v = hg['n_v']
    perm = rng.permutation(n_v)          # old label v -> new label perm[v]
    inv = np.argsort(perm)                # new label -> old label, for feature lookup

    new_members = [sorted(int(perm[v]) for v in mem) for mem in hg['members']]
    new_incident = [[] for _ in range(n_v)]
    for ei, mem in enumerate(new_members):
        for v in mem:
            new_incident[v].append(ei)

    new_x_v = hg['x_v'][inv]              # new_x_v[new_label] = old x_v[old_label]

    return {'members': new_members, 'incident': new_incident, 'n_v': n_v,
            'n_e': hg['n_e'], 'x_v': new_x_v, 'x_e': hg['x_e'].copy()}


def random_pool(n_v, d, pool_size, rng, **kwargs):
    """A pool of `pool_size` independently random hypergraphs, all of
    exactly n_v vertices (order n := n_v, so every pool member lies in
    H_{d,n}^bool). Independent random draws are what give the pool
    structural diversity -- deliberately not deduplicated up front, since
    HWL colour grouping (build_families below) does that filtering."""
    return [random_hypergraph(n_v, d, rng, **kwargs) for _ in range(pool_size)]


def compute_hwl_init(hgs):
    """Round-0 colours for a pool of hypergraphs, using GLOBAL colour
    maps (shared across every hypergraph in the pool) so identical
    colours in different hypergraphs get the same integer id."""
    nc_map, ec_map = {}, {}
    all_nc, all_ec = [], []
    for hg in hgs:
        nc = []
        for row in hg['x_v']:
            key = tuple(row.tolist())
            if key not in nc_map:
                nc_map[key] = len(nc_map)
            nc.append(nc_map[key])
        all_nc.append(nc)

        ec = []
        for i in range(hg['n_e']):
            key = (len(hg['members'][i]), tuple(hg['x_e'][i].tolist()))
            if key not in ec_map:
                ec_map[key] = len(ec_map)
            ec.append(ec_map[key])
        all_ec.append(ec)
    return all_nc, all_ec


def compute_hwl_round(hgs, all_nc, all_ec):
    """One HWL refinement round (hyperedge update, then vertex update,
    using the freshly updated hyperedge colours -- same two-stage order
    as eq:hwl-edge/eq:hwl-vertex and every other HWL implementation in
    this project), with GLOBAL colour maps shared across the pool."""
    new_ec_map = {}
    new_all_ec = []
    for hi, hg in enumerate(hgs):
        new_ec = []
        for ei, mem in enumerate(hg['members']):
            k = (all_ec[hi][ei], tuple(sorted(all_nc[hi][v] for v in mem)))
            if k not in new_ec_map:
                new_ec_map[k] = len(new_ec_map)
            new_ec.append(new_ec_map[k])
        new_all_ec.append(new_ec)

    new_nc_map = {}
    new_all_nc = []
    for hi, hg in enumerate(hgs):
        new_nc = []
        for v in range(hg['n_v']):
            k = (all_nc[hi][v], tuple(sorted(new_all_ec[hi][e] for e in hg['incident'][v])))
            if k not in new_nc_map:
                new_nc_map[k] = len(new_nc_map)
            new_nc.append(new_nc_map[k])
        new_all_nc.append(new_nc)

    return new_all_nc, new_all_ec


def compute_hwl_stable(hgs, n_rounds):
    """Runs exactly n_rounds of HWL refinement from scratch and returns
    the final (all_nc, all_ec). n_rounds should be the pool's order n, so
    the returned colouring is guaranteed stable for every member."""
    all_nc, all_ec = compute_hwl_init(hgs)
    for _ in range(n_rounds):
        all_nc, all_ec = compute_hwl_round(hgs, all_nc, all_ec)
    return all_nc, all_ec


def full_signature(nc, ec):
    """The stable HWL colouring of a single hypergraph, as an order-
    independent, hashable object: the sorted multiset of vertex colours
    paired with the sorted multiset of hyperedge colours -- exactly the
    "stable pair" the paper's HWL definition (sec:hwl) calls the HWL
    colouring of H."""
    return (tuple(sorted(nc)), tuple(sorted(ec)))


def build_families(hgs, all_nc, all_ec):
    """Partitions the pool by stable HWL signature. Returns:
      base_indices: one representative index per distinct signature --
        the maximal pairwise-HWL-distinguishable family found in the pool.
      twin_idx: the index of the first pool member found sharing its
        signature with an existing base-family member (None if every
        pool member turned out pairwise distinguishable, which would
        itself be a notable, if unlikely, finding worth flagging).
      twin_partner_pos: the POSITION within base_indices (not the raw
        pool index) of the base-family member twin_idx duplicates --
        i.e. base_indices[twin_partner_pos] and twin_idx share a stable
        HWL colour, so lem:hwl-upper forces them to receive identical
        HNN_mlp output regardless of Theta. Any random label assignment
        that gives these two positions DIFFERENT target labels is
        provably unshatterable by any width; this is what
        run_exact_bound_experiment.py checks the augmented family
        against directly.
    """
    seen = {}
    pos_of = {}
    base_indices = []
    twin_idx = None
    twin_partner_pos = None
    for i in range(len(hgs)):
        sig = full_signature(all_nc[i], all_ec[i])
        if sig not in seen:
            seen[sig] = i
            pos_of[sig] = len(base_indices)
            base_indices.append(i)
        elif twin_idx is None:
            twin_idx = i
            twin_partner_pos = pos_of[sig]
    return base_indices, twin_idx, twin_partner_pos


def batch_hypergraphs(hgs, feat_dim):
    """Pad a list of hypergraphs (possibly different n_e, but here always
    the same n_v since random_pool fixes order = n_v) into dense batched
    tensors (x, B, node_mask, he_mask), same convention as q1/q2/q3's own
    collate/batch_hypergraphs. Vertex features only (x_v) are fed to the
    model, matching HNN's h_v^0 := x_v initial condition; x_e is used
    solely for HWL colouring (data.py above), not as a model input."""
    import torch
    max_nv = max(hg['n_v'] for hg in hgs)
    max_ne = max(hg['n_e'] for hg in hgs)
    bs = len(hgs)

    xb = np.zeros((bs, max_nv, feat_dim), np.float32)
    Bb = np.zeros((bs, max_ne, max_nv), np.float32)
    nm = np.zeros((bs, max_nv), np.float32)
    hem = np.zeros((bs, max_ne), np.float32)

    for i, hg in enumerate(hgs):
        nv, ne = hg['n_v'], hg['n_e']
        xb[i, :nv] = hg['x_v']
        nm[i, :nv] = 1.0
        hem[i, :ne] = 1.0
        for ei, mem in enumerate(hg['members']):
            for v in mem:
                Bb[i, ei, v] = 1.0

    return (torch.from_numpy(xb), torch.from_numpy(Bb),
            torch.from_numpy(nm), torch.from_numpy(hem))
