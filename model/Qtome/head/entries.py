"""Head entry types: one class per features[] name (ENTRY_TYPES). An entry owns its validation (check), its feature width (dim), whether it needs the trial length in patches (needs_patches) and its forward. `source` says what it reads: 'atom' = the spatially mixed Q-atom codes (a, b), each [B, N', K, S]; 'raw' = the spatially mixed raw patches [B, K, N', L]; 'latent' = z. `e` is the entry's effective config (config._entry_cfg). Parameter/buffer names and construction order are what saved heads were trained with: keep them when changing a class. A new head feature = one class + one ENTRY_TYPES line."""
import torch
import torch.nn as nn

from model.Qtome.patches import overlap_add_patches
from model.Qtome.head.pooling import _window_patches, _time_pool, _window, _pooled_dim


BANDS = ((8.0, 13.0), (13.0, 30.0))   # mu, beta


class _Entry(nn.Module):
    source = 'atom'

    def __init__(self, e):
        super().__init__()
        self.e = e

    @staticmethod
    def check(e):
        pass

    @staticmethod
    def needs_patches(e):
        return e['time_pool'] in ('learned', 'none')


class AtomPowerEntry(_Entry):
    """log of time-pooled a^2 + b^2 per Q-atom: phase-invariant power (induced activity)."""
    def __init__(self, e):
        super().__init__(e)
        self.time = _time_pool(e, e['num_atoms'])

    @staticmethod
    def dim(e, K):
        return _pooled_dim(e, K, e['num_atoms'])

    def forward(self, a, b):
        a, b = _window(self.e, a, b)
        return torch.log(self.time(a.pow(2) + b.pow(2)) + 1e-12).flatten(1)


class RawBandEntry(_Entry):
    """FFT band power (BANDS) of the mixed raw patches."""
    source = 'raw'

    def __init__(self, e):
        super().__init__(e)
        self.time = _time_pool(e, len(BANDS))

    @staticmethod
    def dim(e, K):
        return _pooled_dim(e, K, len(BANDS))

    def forward(self, x):
        e = self.e
        if e['time_pool'] == 'window':
            x = x[:, :, _window_patches(e, x.shape[2], x.device)]
        sp = torch.fft.rfft(x, dim=-1).abs().pow(2)                                 # [B, K, N, bins]
        fr = torch.fft.rfftfreq(x.shape[-1], 1.0 / e['sample_freq']).to(sp.device)
        p = torch.stack([sp[..., (lo <= fr) & (fr < hi)].sum(-1) for lo, hi in BANDS], -1)  # [B, K, N, 2]
        return torch.log(self.time(p.permute(0, 2, 1, 3)) + 1e-12).flatten(1)


class RawSignalEntry(_Entry):
    """The mixed raw signal itself, overlap-added and average-pooled to ~20 Hz."""
    source = 'raw'

    def __init__(self, e):
        super().__init__(e)
        self.pool_k = max(1, round(e['sample_freq'] / 20))

    @staticmethod
    def check(e):
        if e['time_pool'] != 'none':
            raise ValueError(f"features entry 'raw_signal' requires effective time_pool='none', got {e['time_pool']!r}")

    @staticmethod
    def needs_patches(e):
        return True

    @staticmethod
    def dim(e, K):
        pool = max(1, round(e['sample_freq'] / 20))
        return K * (((e['num_patches'] - 1) * e['patch_stride'] + e['patch_len']) // pool)

    def forward(self, x):
        sig = overlap_add_patches(x, self.e['patch_stride'])
        return torch.nn.functional.avg_pool1d(sig, self.pool_k, self.pool_k).flatten(1)


class SignedABEntry(_Entry):
    """signed_ab (2026-09-25): the signed, phase-locked counterpart of atom_power. Fully linear
    and factored: spatially mixed a and b (kept signed, so polarity and phase survive) ->
    learned Q-atom pooling S -> atom_rank -> learned time filters N' -> time_rank (init: flat
    average + small noise). A free linear readout on raw a/b would be the same function class
    but ~10^4 weights per class; the factoring is the regularisation. Carries evoked /
    phase-locked content only: induced (random-phase) power averages out, pair it with
    atom_power for that. -> [B, 2 * K * atom_rank * time_rank]."""
    def __init__(self, e):
        super().__init__(e)
        M, R, N = int(e['atom_rank']), int(e['time_rank']), e['num_patches']
        self.atom = nn.Linear(e['num_atoms'], M, bias=False)
        self.q = nn.Parameter(torch.full((R, N), 1.0 / N) + torch.randn(R, N) * 0.02)

    @staticmethod
    def check(e):
        if int(e['atom_rank']) < 1 or int(e['time_rank']) < 1:
            raise ValueError("features entry 'signed_ab' requires atom_rank >= 1 and time_rank >= 1")

    @staticmethod
    def needs_patches(e):
        return True

    @staticmethod
    def dim(e, K):
        return 2 * K * int(e['atom_rank']) * int(e['time_rank'])

    def forward(self, a, b):
        return torch.cat([torch.einsum('rn,bnkm->bkmr', self.q, self.atom(t)) for t in (a, b)], dim=1).flatten(1)


def _latent_proj(e, m):
    """z's D -> m projection. latent_proj 'learned': trained with the head. 'pca': frozen here, set
    by train_finetune.py's run_one to the top-m principal axes of the run's TRAINING-trial z
    (no trained parameters, like the Q-atom head's fixed templates)."""
    if e['latent_proj'] not in ('learned', 'pca'):
        raise ValueError(f"latent_proj must be learned|pca, got {e['latent_proj']!r}")
    proj = nn.Linear(e['latent_dim'], m, bias=False)
    proj.weight.requires_grad_(e['latent_proj'] == 'learned')
    return proj


class LatentSignedEntry(_Entry):
    """The signed_ab pipeline on z: learned projection D -> 2 * atom_rank (signed_ab has a and b
    at atom_rank each, so the width matches), learned time filters -> time_rank."""
    source = 'latent'

    def __init__(self, e):
        super().__init__(e)
        M, R, N = 2 * int(e['atom_rank']), int(e['time_rank']), e['num_patches']
        self.proj = _latent_proj(e, M)
        self.q = nn.Parameter(torch.full((R, N), 1.0 / N) + torch.randn(R, N) * 0.02)

    check = SignedABEntry.check
    needs_patches = SignedABEntry.needs_patches

    @staticmethod
    def dim(e, K):
        return SignedABEntry.dim(e, K)

    def forward(self, z):
        return torch.einsum('rn,bnkm->bkmr', self.q, self.proj(z)).flatten(1)


ENTRY_TYPES = {'atom_power': AtomPowerEntry, 'signed_ab': SignedABEntry, 'latent_signed': LatentSignedEntry,
               'raw_band': RawBandEntry, 'raw_signal': RawSignalEntry}


FEATURES_ALL = tuple(ENTRY_TYPES)
