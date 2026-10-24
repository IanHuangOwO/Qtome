"""Electrode-position and time embeddings: Fourier features of the coordinates, the directional relative spatial bias, FoPE and the patch embedding."""
import math

import torch
import torch.nn as nn


# Fixed log-spaced wavelengths for electrode positions in metres (a head is ~0.2 m across):
# 40 cm ~ hemisphere scale down to 2.5 cm ~ a dense cap's electrode spacing.
FOURIER_WAVELENGTHS_M = (0.40, 0.20, 0.10, 0.05, 0.025)


def fourier_features(p):
    """[..., 3] positions (m) -> [..., 3 + 3*2*len(FOURIER_WAVELENGTHS_M)]: raw xyz plus sin/cos
    of every axis at every wavelength. Gives an MLP a multi-scale basis over the scalp, which
    a raw-xyz MLP lacks (measured: it drifted to a near-constant, channel-independent output)."""
    w = 2 * math.pi / torch.tensor(FOURIER_WAVELENGTHS_M, device=p.device, dtype=p.dtype)   # [F]
    ang = (p.unsqueeze(-1) * w).flatten(-2)                                                 # [..., 3F]
    return torch.cat([p, ang.sin(), ang.cos()], dim=-1)


FOURIER_DIM = 3 + 3 * 2 * len(FOURIER_WAVELENGTHS_M)


class RelativeSpatialBias(nn.Module):
    """Per-block, per-head bias added to spatial attention scores from the DIRECTIONAL
    position difference of the two electrodes: b(i, j) = MLP(fourier(p_i - p_j)), so the model
    can learn asymmetric relations (posterior vs anterior, left vs right), not only distance.
    Depends only on the montage, so it's computed once per window and shared over time.
    Last layer zero-initialised: training starts exactly as without it. -> [B, depth, H, C, C]."""
    def __init__(self, depth, num_heads, hidden=32):
        super().__init__()
        self.depth, self.heads = depth, num_heads
        out = nn.Linear(hidden, depth * num_heads)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.mlp = nn.Sequential(nn.Linear(FOURIER_DIM, hidden), nn.GELU(), out)

    def forward(self, coords):
        B, C, _ = coords.shape
        rel = coords[:, :, None, :] - coords[:, None, :, :]                                   # [B, C, C, 3]  i - j
        b = self.mlp(fourier_features(rel.float()))                                          # [B, C, C, depth*H]
        return b.view(B, C, C, self.depth, self.heads).permute(0, 3, 4, 1, 2)                # [B, depth, H, C, C]


class FoPE(nn.Module):
    """Fourier Position Embedding (Hua et al., ICML 2025): RoPE on temporal attention's q and k in which every
    frequency slot is a fixed Fourier series of all the frequencies (identity + N(0, sigma) mixing, separate
    for cos and sin and per head, not trained) and frequencies whose period exceeds the training length are
    clipped to zero. Frequencies: head_dim/2 periods spaced geometrically from 2 to 2 * train_len fine patches
    (RoPE's base-10000 spacing would leave 2 of 10 below the clip at our 19-patch windows). Positions are in
    fine patches, a pooled token at its patches' mean, so a lag means the same time at every stage. The only time
    position since 2026-10-07 (it replaced the learned absolute time table + RelativeTemporalBias: level masked MSE
    at 5 s, inputs 2x / 4x the training length degrade x1.00-1.11 instead of up to x2.72). pos [n] -> (cos, sin) [H, n, head_dim]."""
    def __init__(self, num_heads, head_dim, train_len=19, sigma=0.3):
        super().__init__()
        assert head_dim % 2 == 0, head_dim
        m = head_dim // 2
        tau = torch.logspace(math.log10(2.0), math.log10(2.0 * train_len), m)
        omega = 2 * math.pi / tau
        omega[tau > train_len] = 0.0                     # under-trained: never a full period in training
        self.register_buffer('omega', omega)
        std = sigma * math.sqrt(2.0 / (m + m))           # Xavier-normal std at gain sigma, as the reference code
        self.register_buffer('cos_coef', torch.randn(num_heads, m, m) * std + torch.eye(m))
        self.register_buffer('sin_coef', torch.randn(num_heads, m, m) * std + torch.eye(m))

    def forward(self, pos):
        ang = pos[:, None] * self.omega                                                      # [n, m]
        cos = torch.einsum('nm,hmk->hnk', ang.cos(), self.cos_coef)
        sin = torch.einsum('nm,hmk->hnk', ang.sin(), self.sin_coef)
        return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)                          # [H, n, head_dim]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class SpatialTemporalEmbeddings(nn.Module):
    def __init__(self, patch_len, dim, spatial=True):
        super().__init__()
        self.proj = nn.Linear(patch_len, dim)
        self.norm = nn.LayerNorm(dim)
        # No absolute time term: time enters temporal attention through FoPE (TSAEncoder).
        self.spatial_active = False
        # spatial=False: no coordinate embedding at all (the spatial-embedding ablation; the model
        # then pairs it with no RelativeSpatialBias either, see QtomePretrain).
        # bias=False on BOTH linears: a bias on either one is a channel-INDEPENDENT constant the
        # network can add regardless of coords (it collapsed to exactly that before); with no bias
        # anywhere, per-channel variation is the only thing this path can produce.
        self.coord_proj = None
        if spatial:
            _coord_out = nn.Linear(dim // 4, dim, bias=False)
            nn.init.zeros_(_coord_out.weight)
            self.coord_proj = nn.Sequential(nn.Linear(FOURIER_DIM, dim // 4, bias=False), nn.GELU(), _coord_out)

    def enable_spatial(self):
        self.spatial_active = True

    def forward(self, x, coords=None, bool_masked_pos=None, mask_token=None):
        B, C, N, L = x.shape
        z = self.proj(x.reshape(B * C, N, L))  # [B*C, N, D]
        if bool_masked_pos is not None:
            # masked content -> mask_token BEFORE the position terms below (see QtomePretrain)
            z = torch.where(bool_masked_pos.reshape(B * C, N, 1), mask_token.reshape(1, 1, -1).to(z.dtype), z)

        if coords is not None and self.spatial_active and self.coord_proj is not None:
            s = self.coord_proj(fourier_features(coords).reshape(B * C, -1)).unsqueeze(1)  # [B*C, 1, D]
            z = z + s

        return self.norm(z).reshape(B, C, N, -1)
