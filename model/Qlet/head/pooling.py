"""Finetune head, spatial and time pooling. Shapes: a, b are the spatially mixed code amplitudes [B, N', K, S] (B trials, N' patches, K spatial filters, S qlets); power = a^2 + b^2. Parameter names (p, q) and init match the old QletFeatureHead so saved checkpoints load unchanged."""
import math

import torch
import torch.nn as nn


def spatial_mix(spatial, t, dim):
    """Signed spatial filter (nn.Linear(C, K, bias=False)) over channel axis `dim`, or
    identity when spatial is None (channel concat)."""
    if spatial is None:
        return t
    return torch.movedim(spatial(torch.movedim(t, dim, -1).float()), -1, dim)


class PerAtomSpatial(nn.Module):
    """One signed spatial filter bank per qlet (filter-bank-CSP style): weight [S, K, C], so qlet s's code is mixed
    by its own K filters. [B, N', C, S] -> [B, N', K, S], the same shape the shared nn.Linear(C, K) gives.
    Init as nn.Linear's (uniform, bound 1/sqrt(C))."""
    def __init__(self, num_channels, k, num_atoms):
        super().__init__()
        b = 1.0 / math.sqrt(num_channels)
        self.weight = nn.Parameter(torch.empty(num_atoms, k, num_channels).uniform_(-b, b))

    def forward(self, t):
        return torch.einsum('bncs,skc->bnks', t.float(), self.weight)


class FlatTimePool(nn.Module):
    """Uniform weights over patches."""
    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return power.mean(1)


class LearnedTimePool(nn.Module):
    """Low-rank softmax time weights w[s, n] = softmax_n(sum_r p[r, s] q[r, n]).
    Small init => starts equal to the flat mean."""
    def __init__(self, rank, num_atoms, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_atoms) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def weights(self):                                           # [S, N']
        return torch.softmax(torch.einsum('rs,rn->sn', self.p, self.q), dim=-1)

    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return torch.einsum('sn,bnks->bks', self.weights(), power)


class NoTimePool(nn.Module):
    """Keep the patch axis: the features stay [B, N', K, F] and are flattened by the head."""
    def forward(self, power):
        return power


def _window_patches(cfg, N, device):
    """Bool mask [N] of patches fully inside cfg['window'] seconds (same rule as before)."""
    starts = torch.arange(N, device=device) * cfg['patch_stride']
    lo, hi = (w * cfg['sample_freq'] for w in cfg['window'])
    keep = (starts >= lo) & (starts + cfg['patch_len'] <= hi)
    assert keep.any(), f"window {cfg['window']} s keeps no patch"
    return keep


def _time_pool(e, n_feat):
    if e['time_pool'] == 'learned':
        return LearnedTimePool(int(e['time_rank']), n_feat, e['num_patches'])
    return NoTimePool() if e['time_pool'] == 'none' else FlatTimePool()


def _window(e, a, b):
    if e['time_pool'] != 'window':
        return a, b
    m = _window_patches(e, a.shape[1], a.device)
    return a[:, m], b[:, m]


def _pooled_dim(e, K, n_feat):
    return (e['num_patches'] if e['time_pool'] == 'none' else 1) * K * n_feat
