"""HNN_slp architecture (hnn-vc.pdf Eq. 1-2) -- identical two-stage
edge-then-node update used in q1_width_sweep.py / q2_depth_sweep.py. The
readout is out_dim=1 (matching the theorem's scalar readout exactly,
d+1 params) and h_v^(0)=x_v is zero-padded up to width d rather than
learned, so count_params() equals param_count_formula(d, n_layers) --
the theorem's own P_H = d(4dL+2L+1)+1 from the/hnn_vc_full_proof.tex --
byte-for-byte, asserted at construction time. Training loop differs from
Q1/Q2: BCEWithLogitsLoss and fit() trains to fit a FIXED given label
assignment, stopping early once every example is correctly classified
(used to test shattering: can HNN_slp(d,L) fit every one of many random
labelings of a hypergraph family of a given size?)."""
import torch
import torch.nn as nn
from tqdm import trange


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
    def __init__(self, feat_dim, d, n_layers, out_dim=1):
        super().__init__()
        assert feat_dim <= d, f"feat_dim={feat_dim} must be <= hidden width d={d} to zero-pad"
        self.d = d
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList([HNNLayer(d) for _ in range(n_layers)])
        self.readout = nn.Linear(d, out_dim)
        n = self.count_params()
        formula = param_count_formula(d, n_layers, out_dim)
        assert n == formula, f"actual param count {n} != closed-form P_H {formula}"

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

    def fit(self, x, B, node_mask, he_mask, y, train_steps, lr=1e-4,
            device='cpu', tqdm_prefix=''):
        """Train to fit the fixed label vector y. Returns the best
        training accuracy reached (1.0 == this label assignment was
        shattered by this (d, L) configuration)."""
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


def param_count_formula(d, n_layers, out_dim=1):
    """Closed-form parameter count, exactly the/hnn_vc_full_proof.tex's
    P_H = d(4dL+2L+1)+1 (out_dim=1 case); generalises the readout term
    d+1 to out_dim*(d+1) for out_dim>1. The zero-padded input has no
    trainable parameters, so every other term is unchanged."""
    return 4 * n_layers * d**2 + 2 * n_layers * d + out_dim * (d + 1)
