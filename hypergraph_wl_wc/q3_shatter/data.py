"""Hyper-TNM hypergraph family -- hypergraph analogue of Morris et al.'s
TNM tree family (wl_vs_vc/Q3/data.py: tnm_tree / get_tnm_trees).

Original TNM tree (plain graph): a depth-2 rooted tree -- root r, two
depth-1 "hub" nodes h1/h2 each attached to r by a single edge, and
deg_1/deg_2 depth-2 leaves attached to h1/h2 respectively by individual
edges (deg_1 + deg_2 = order).

Hyper-TNM hypergraph (this file): identical enumeration and node count,
but each hub's leaves are grouped into a SINGLE hyperedge with that hub,
instead of deg_1 (resp. deg_2) separate pairwise edges. This is the part
that is genuinely hypergraph-specific: cardinality varies with deg_1/deg_2
(not just node degree), which a two-stage HWL/HNN_slp model can exploit
directly but which collapses to plain node-degree information under a
1-WL/GNN clique-expansion of the same structure.
"""
import numpy as np
import torch


def hyper_tnm(deg_1, deg_2):
    """One hyper-TNM hypergraph. Nodes: r=0, h1=1, h2=2, leaves=3..n_v-1.
    Hyperedges: {r,h1,h2} (card 3), {h1}+leaves[:deg_1] (card deg_1+1,
    omitted if deg_1==0), {h2}+leaves[deg_1:] (card deg_2+1, omitted if
    deg_2==0)."""
    n_v = 3 + deg_1 + deg_2
    leaves = list(range(3, n_v))

    members = [[0, 1, 2]]
    if deg_1 > 0:
        members.append([1] + leaves[:deg_1])
    if deg_2 > 0:
        members.append([2] + leaves[deg_1:])

    return {'members': members, 'n_v': n_v}


def get_hyper_tnm_hypergraphs(order):
    """order = deg_1 + deg_2 (total leaf count). Mirrors
    wl_vs_vc/Q3/data.py's get_tnm_trees(order) exactly: returns
    floor(order/2)+1 structurally distinct hypergraphs of the same node
    count (3+order), ranging from a maximally lopsided split (deg_1=0,
    deg_2=order) to a balanced split (deg_1==deg_2)."""
    max_idx = order // 2
    return [hyper_tnm(i, order - i) for i in range(max_idx + 1)]


def batch_hypergraphs(hgs):
    """Pad a list of same-order hypergraphs into dense batched tensors
    (x, B, node_mask, he_mask), same convention as collate() in
    q1_width_sweep.py / q2_depth_sweep.py. Node feature is a constant 1.0
    (matches tnm_tree's x = torch.ones(order) -- Q3 tests architectural
    capacity to fit arbitrary labels, not feature-driven learning)."""
    max_nv = max(h['n_v'] for h in hgs)
    max_ne = max(len(h['members']) for h in hgs)
    bs = len(hgs)

    xb = np.zeros((bs, max_nv, 1), np.float32)
    Bb = np.zeros((bs, max_ne, max_nv), np.float32)
    nm = np.zeros((bs, max_nv), np.float32)
    hem = np.zeros((bs, max_ne), np.float32)

    for i, h in enumerate(hgs):
        nv = h['n_v']
        xb[i, :nv, 0] = 1.0
        nm[i, :nv] = 1.0
        for ei, mem in enumerate(h['members']):
            for v in mem:
                Bb[i, ei, v] = 1.0
            hem[i, ei] = 1.0

    return (torch.from_numpy(xb), torch.from_numpy(Bb),
            torch.from_numpy(nm), torch.from_numpy(hem))
