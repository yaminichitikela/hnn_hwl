"""
HNN_mlp architecture for the prop:upper-nonuniform exact-bound test.

Every other experiment in this project (Q1/Q2/Q3) uses HNN_slp -- single-
layer-perceptron update and readout, the restricted class whose parameter
count P the VC-dimension-via-parameters bound (thm:vc-hnn) needs
explicitly. The exact bound tested here (Proposition "Upper and matching
lower bound", VCdim_{H_{d,n}^bool}(HNN_mlp(L)) = m^H_{n,d,L}) is instead
stated for the more general HNN_mlp class: sum aggregation (fixed, not a
free architectural choice -- see the paper's own HNN_mlp(d,L) definition
in Preliminaries), but MLP -- not single-layer -- update and readout
networks, needed for the lower-bound (shattering) construction to have
enough representational freedom to simulate HWL's injective colour
refinement exactly (lem:hnn-simulates-hwl).
"""
import torch
import torch.nn as nn
from tqdm import trange


class HNNmlpLayer(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.vertex_mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, h_v, h_e, B, node_mask, he_mask):
        Bh_v = torch.bmm(B, h_v)                                    # sum aggregation: member vertices -> hyperedge
        h_e = self.edge_mlp(torch.cat([h_e, Bh_v], dim=-1)) * he_mask.unsqueeze(-1)
        BTh_e = torch.bmm(B.transpose(1, 2), h_e)                   # sum aggregation: incident hyperedges -> vertex
        h_v = self.vertex_mlp(torch.cat([h_v, BTh_e], dim=-1)) * node_mask.unsqueeze(-1)
        return h_v, h_e


class HNNmlpClassifier(nn.Module):
    """h_v^(0) = x_v, zero-padded (not learned) up to hidden width d,
    same convention as HNN_slp elsewhere in this project. `hidden`
    controls the width of the single ReLU hidden layer inside every
    update/readout MLP; defaults to d itself (width-d update MLPs,
    width-d readout MLP) if not given explicitly."""
    def __init__(self, feat_dim, d, n_layers, hidden=None):
        super().__init__()
        assert feat_dim <= d, f"feat_dim={feat_dim} must be <= hidden width d={d} to zero-pad"
        self.d = d
        self.feat_dim = feat_dim
        h = hidden or d
        self.layers = nn.ModuleList([HNNmlpLayer(d, h) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, B, node_mask, he_mask):
        pad = x.new_zeros(*x.shape[:-1], self.d - self.feat_dim)
        h_v = torch.cat([x, pad], dim=-1) * node_mask.unsqueeze(-1)
        h_e = torch.zeros(B.shape[0], B.shape[1], self.d, device=x.device)
        for layer in self.layers:
            h_v, h_e = layer(h_v, h_e, B, node_mask, he_mask)
        h_G = (h_v * node_mask.unsqueeze(-1)).sum(dim=1)
        return self.readout(h_G)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def fit(self, x, B, node_mask, he_mask, y, train_steps, lr=1e-3,
            device='cpu', tqdm_prefix=''):
        """Train to fit the fixed label vector y (one label per
        hypergraph in the batch, i.e. per family member). Returns the
        best training accuracy reached (1.0 == this label assignment was
        shattered by this configuration). Same protocol as
        q3_shatter/model.py's fit(): BCEWithLogitsLoss, early stop the
        moment every family member is classified correctly."""
        self.to(device)
        x, B, node_mask, he_mask, y = (
            t.to(device) for t in (x, B, node_mask, he_mask, y)
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        crit = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        self.train()
        with trange(train_steps) as t:
            t.set_description(tqdm_prefix)
            for _ in t:
                opt.zero_grad()
                logits = self(x, B, node_mask, he_mask).view(-1)
                loss = crit(logits, y.float())
                loss.backward()
                opt.step()

                pred = (logits > 0).long()
                correct = (pred == y)
                acc = correct.float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                t.set_postfix(loss=loss.item(), acc=acc)

                if bool(correct.all()):
                    return 1.0
        return best_acc
