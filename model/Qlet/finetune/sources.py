"""Where a finetune head reads its trials from: the qlet code cache, the raw compiled data, or both, as one source with get(idx)."""
import copy

import torch

from IO.dataset import build_dataset_from_config
from IO.preprocessing import num_patches, slice_patches
from model.Qlet.finetune.feature_cache import CachedAtomDataset, get_atom_cache
from model.Qlet.head.config import feature_names, needs_atom, needs_latent, needs_raw


class AtomSource:
    """Cached qlet amplitudes of the pool (feature_cache.py), moved to the device once (a
    BNCI2014008 64-channel impute cache is ~1 GB fp16): the head is tiny, so per-batch CPU
    indexing and host-to-device copies were most of a step's time."""
    kind = 'atom'

    def __init__(self, config, ds_name, pool, device, latent=False):
        subs = [str(s) for s in pool]
        self.data = CachedAtomDataset(get_atom_cache(config, ds_name, subs, latent=latent), subs)
        self.labels, self.subject_data = self.data.labels, self.data.subject_data
        self.channel_idx = self.data.channel_idx
        self.num_patches, self.num_atoms = self.data.num_patches, self.data.num_atoms
        self.amp, self.labels_dev = self.data.amp.to(device), self.labels.to(device)
        self.z = self.data.z.to(device) if latent else None                 # [T, N', Cv, D] fp16
        self.latent_dim = self.z.shape[-1] if latent else None

    def get(self, idx):
        idx = idx.to(self.amp.device)
        out = {'atom': self.amp[idx].float()}
        if self.z is not None:
            out['latent'] = self.z[idx].float()
        return out, self.labels_dev[idx]


class RawSource:
    """Compiled raw trials of the pool, real channels only; patched on the fly."""
    kind = 'raw'

    def __init__(self, config, ds_name, pool):
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds_name: {**config['dataset_params']['finetune'][ds_name],
                                                       'subject_to_use': list(pool)}}
        base = build_dataset_from_config(cfg, mode='finetune').base_dataset
        assert len({tuple(v.tolist()) for v in base.all_valid_channels}) == 1, "one real-channel set per dataset"
        self.channel_idx = torch.nonzero(base.all_valid_channels[0]).flatten().tolist()
        self.x = base.data[:, self.channel_idx].contiguous()                  # [N, C_valid, T]
        self.labels, self.subject_data = base.labels.long(), base.subject_data.long()
        pp = config['preprocess_params']
        self.patch_len = pp.get('patch_length', 100)
        self.patch_stride = pp.get('patch_stride', self.patch_len)
        self.num_patches = num_patches(self.x.shape[-1], self.patch_len, self.patch_stride)
        self.num_atoms = 0

    def get(self, idx):
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)  # [B, C_valid, N', L]
        return {'raw': xp}, self.labels[idx]


class CombinedSource:
    """Serves a AtomSource and a RawSource together, for a head whose features list needs
    both (e.g. features=['atom_power', 'raw_band']). Exposes the union of attributes either
    single source exposes (num_patches/num_atoms/channel_idx/labels/subject_data) --
    both sources are built from the SAME (ds_name, pool), so their per-trial ordering,
    labels and channel_idx must already agree; asserted once at construction, not re-checked
    per batch."""
    kind = 'combined'

    def __init__(self, atom_source, raw_source):
        assert atom_source.channel_idx == raw_source.channel_idx, \
            "AtomSource/RawSource channel_idx mismatch -- same dataset/pool should agree"
        assert torch.equal(atom_source.labels, raw_source.labels), \
            "AtomSource/RawSource label order mismatch -- same dataset/pool should agree"
        self.atom, self.raw = atom_source, raw_source
        self.labels, self.subject_data = atom_source.labels, atom_source.subject_data
        self.channel_idx = atom_source.channel_idx
        self.num_patches, self.num_atoms = atom_source.num_patches, atom_source.num_atoms
        self.z, self.latent_dim = atom_source.z, atom_source.latent_dim

    def get(self, idx):
        atom_d, y = self.atom.get(idx)
        raw_d, _ = self.raw.get(idx)
        return {**atom_d, **raw_d}, y


def make_source(config, ds_name, pool, device):
    ft_cfg = dict(config['model_params']['Qlet']['finetune'])
    want_atom, want_raw, latent = needs_atom(ft_cfg), needs_raw(ft_cfg), needs_latent(ft_cfg)
    if latent:   # latent_source: which z the latent_* entries read ('output' = what the qlets read)
        latent = ft_cfg.get('latent_source', 'output')
        if latent not in ('output', 'bottleneck'):
            raise ValueError(f"latent_source must be output|bottleneck, got {latent!r}")
    k = int(config['training_params']['finetune'].get('latent_pool', 1))
    if k > 1:   # experiment (2026-09-29): average k adjacent z tokens, for a head that reads only z
        if not all(n.startswith('latent_') for n in feature_names(ft_cfg)):
            raise ValueError("training_params.finetune.latent_pool needs a head with only latent_* entries")
        src = AtomSource(config, ds_name, pool, device, latent)
        src.z = pool_tokens(src.z, k)
        src.num_patches = src.z.shape[1]
        return src
    if want_atom and want_raw:
        return CombinedSource(AtomSource(config, ds_name, pool, device, latent), RawSource(config, ds_name, pool))
    if want_atom:
        return AtomSource(config, ds_name, pool, device, latent)
    return RawSource(config, ds_name, pool)


def pool_tokens(z, k):
    """z [T, N, ...] -> [T, N // k, ...]: mean of k adjacent tokens (a trailing N % k is dropped). Splits only the
    token axis, so the reshape is safe (docs/agents/reshape-pitfalls.md)."""
    n = z.shape[1] // k * k
    return z[:, :n].reshape(z.shape[0], n // k, k, *z.shape[2:]).float().mean(2).to(z.dtype)


def iter_batches(source, idx, batch_size, device, shuffle, gen=None):
    idx = torch.as_tensor(idx, dtype=torch.long)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen)]
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        if shuffle and len(j) < 2:
            continue
        x, y = source.get(j)
        yield {k: v.to(device) for k, v in x.items()}, y.to(device)
