"""The composable finetune head: one entry per cfg['features'] item, each behind its own spatial filter, concatenated -> BatchNorm / Dropout / Linear readout. Runnable self-check: _selfcheck_head_modules()."""
import torch
import torch.nn as nn

from model.Qlet.head.pooling import spatial_mix, PerAtomSpatial, FlatTimePool, LearnedTimePool
from model.Qlet.head.config import feature_names, resolve_head_config, _entry_cfg, needs_atom, needs_raw, _entry_dim, feature_dim
from model.Qlet.head.entries import ENTRY_TYPES


def _selfcheck_head_modules():
    """Runnable check against independent formulas; not run on import."""
    # self-check against independent formulas (complex dtype, explicit softmax)
    torch.manual_seed(0)
    B, N, K, S, R = 3, 39, 8, 25, 2
    a, b = torch.randn(B, N, K, S), torch.randn(B, N, K, S)
    ltp = LearnedTimePool(R, S, N)
    nn.init.normal_(ltp.p); nn.init.normal_(ltp.q)
    w = ltp.weights()
    assert torch.allclose(w.sum(-1), torch.ones(S), atol=1e-6)
    power = a.pow(2) + b.pow(2)
    ref = torch.einsum('sn,bnks->bks', w, power)
    assert torch.allclose(ltp(power), ref, atol=1e-6)
    nn.init.zeros_(ltp.p)                                        # zero logits => uniform => flat mean
    assert torch.allclose(ltp(power), FlatTimePool()(power), atol=1e-5)
    lin = nn.Linear(64, K, bias=False)
    t = torch.randn(B, N, 64, S)
    assert spatial_mix(lin, t, 2).shape == (B, N, K, S) and spatial_mix(None, t, 2) is t
    assert [n for n, _ in ltp.named_parameters()] == ['p', 'q']

    # -- head config: one form, features = [{"type": ...}] with top-level defaults --
    base = dict(num_channels=22, num_atoms=25, sample_freq=200.0, patch_len=50, patch_stride=50)
    two = resolve_head_config(dict(features=[{'type': 'atom_power'}, {'type': 'raw_band', 'time_pool': 'flat'}],
                                   spatial_k=8, time_pool='learned', time_rank=2), num_patches=16, **base)
    assert needs_atom(two) and needs_raw(two) and feature_names(two) == ['atom_power', 'raw_band']
    assert _entry_cfg(two, 'atom_power')['time_pool'] == 'learned' and _entry_cfg(two, 'raw_band')['time_pool'] == 'flat'
    assert feature_dim(two) == _entry_dim(two, 'atom_power') + _entry_dim(two, 'raw_band')
    for bad in (dict(features=['atom_power']), dict(features=[{'type': 'atom_power'}] * 2),
                dict(features=[{'type': 'nope'}]), dict(features=[{'type': 'atom_power', 'typo': 1}]),
                dict(features=[{'type': 'atom_power'}], overrides={})):
        try:
            resolve_head_config(bad, num_patches=16, **base)
            assert False, f"{bad} should have raised"
        except ValueError:
            pass
    hcfg = resolve_head_config(dict(features=[{'type': 'atom_power', 'spatial_k': 4}, {'type': 'raw_band'},
                                              {'type': 'signed_ab'}],
                                    time_pool='flat', dropout=0.0, time_rank=2),
                               num_patches=8, num_channels=6, num_classes=3, num_atoms=S,
                               sample_freq=200.0, patch_len=50, patch_stride=50)
    out = FeatureHead(hcfg)({'atom': torch.randn(2, 8, 6, S, 2), 'raw': torch.randn(2, 6, 8, 50)})
    assert out.shape == (2, 3), out.shape
    ps = resolve_head_config(dict(features=[{'type': 'atom_power', 'spatial_k': 2, 'spatial_per_atom': True}], time_pool='flat',
                                  dropout=0.0), num_patches=8, num_channels=6, num_classes=3, num_atoms=S,
                             sample_freq=200.0, patch_len=50, patch_stride=50)
    ph = FeatureHead(ps)
    x = torch.randn(2, 8, 6, S, 2)
    assert ph({'atom': x}).shape == (2, 3)
    W = ph.spatials['atom_power'].weight                        # [S, K, C]: qlet s mixed by its own filters only
    ref = torch.stack([x[..., s, 0] @ W[s].T for s in range(S)], -1)
    assert torch.allclose(ph.spatials['atom_power'](x[..., 0]), ref, atol=1e-5)

    print('head_modules self-check OK')


class FeatureHead(nn.Module):
    """Composable finetune head, no backbone inside: one ENTRY_TYPES submodule per
    cfg['features'] entry, each behind its own spatial filter, concatenated ->
    BatchNorm/Dropout/Linear readout."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        names = feature_names(cfg)
        self.entries = nn.ModuleDict({name: ENTRY_TYPES[name](_entry_cfg(cfg, name)) for name in names})
        # spatial_k None/0 = no mixing (ablation control): spatial_mix(None, ...) is identity,
        # so each real channel stays its own feature row instead of being pooled to K filters.
        self.spatials = nn.ModuleDict({name: (PerAtomSpatial(cfg['num_channels'], k, cfg['num_atoms'])
                                              if _entry_cfg(cfg, name)['spatial_per_atom']
                                              else nn.Linear(cfg['num_channels'], k, bias=False))
                                       for name in names if (k := _entry_cfg(cfg, name)['spatial_k'])})
        n_feat = feature_dim(cfg)
        # BatchNorm stands in for the probe's StandardScaler: log-powers are far from unit scale.
        self.cls = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(cfg['dropout']),
                                 nn.Linear(n_feat, cfg['num_classes']))

    def forward(self, inp):
        """inp: {'raw': [B, C, N', L] or absent, 'Qlet': [B, N', C, S, 2] or absent}."""
        # Feature math in fp32: under autocast, squared amplitudes overflow in fp16 and the 1e-12
        # epsilon rounds to 0 (NaN loss). The readout stays outside, as before.
        with torch.autocast(device_type=next(iter(inp.values())).device.type, enabled=False):
            raw = inp['raw'].float() if 'raw' in inp else None
            amp = inp['atom'].float() if 'atom' in inp else None
            latent = inp['latent'].float() if 'latent' in inp else None                # [B, N', C, D]
            outs = []
            for name, mod in self.entries.items():
                spatial = self.spatials[name] if name in self.spatials else None
                if mod.source == 'raw':
                    outs.append(mod(spatial_mix(spatial, raw, 1)))                           # [B, K, N', L]
                elif mod.source == 'latent':
                    outs.append(mod(spatial_mix(spatial, latent, 2)))                        # [B, N', K, D]
                elif isinstance(spatial, PerAtomSpatial):
                    outs.append(mod(spatial(amp[..., 0]), spatial(amp[..., 1])))              # each [B, N', K, S]
                else:
                    outs.append(mod(spatial_mix(spatial, amp[..., 0], 2),
                                    spatial_mix(spatial, amp[..., 1], 2)))                  # each [B, N', K, S]
            feat = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return self.cls(feat)
