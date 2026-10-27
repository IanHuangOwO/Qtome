"""Finetune head configuration: defaults, validation (resolve_head_config), the head checkpoint, and the entry widths."""
from model.Qlet.head.entries import ENTRY_TYPES, FEATURES_ALL


# Per-entry keys: set at the top level as the default for every entry, or inside one entry.
_ENTRY_KEYS = ('time_pool', 'time_rank', 'window', 'spatial_k', 'atom_rank', 'latent_proj',
               'spatial_per_atom')


_HEAD_DEFAULTS = dict(features=[{'type': 'atom_power'}], spatial_k=8, time_pool='learned', time_rank=2,
                      window=None, atom_rank=4, latent_proj='learned', spatial_per_atom=False, dropout=0.5,
                      latent_source='output')


def make_head_checkpoint(head, head_cfg, channel_idx, backbone_checkpoint):
    """Head-only checkpoint: state, resolved config (plus the real channels it was built for) and the
    frozen backbone it belongs to. Loaded by FinetuneModel.from_checkpoint."""
    return {'model_state_dict': head.state_dict(),
            'head_config': dict(head_cfg, channel_idx=list(channel_idx)),
            'backbone_checkpoint': backbone_checkpoint}


def feature_names(cfg):
    """The entry types of cfg['features'], in order."""
    return [f['type'] for f in cfg['features']]


def resolve_head_config(ft_params, **derived):
    """Head config = defaults + user keys + derived shapes, validated. features is a list of
    {"type": <ENTRY_TYPES name>, <_ENTRY_KEYS for that entry>}; top-level _ENTRY_KEYS are the
    defaults of every entry. Each entry gets its own spatial filter (its spatial_k, None/0 = no
    mixing, every real channel kept)."""
    user = dict(ft_params)
    unknown = set(user) - set(_HEAD_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown head keys {sorted(unknown)}; valid: {sorted(_HEAD_DEFAULTS)}")
    cfg = {**_HEAD_DEFAULTS, **user, **derived}
    feats = cfg['features']
    if not feats or not all(isinstance(f, dict) and f.get('type') in ENTRY_TYPES for f in feats):
        raise ValueError(f"features must be a non-empty list of {{'type': one of {FEATURES_ALL}, ...}}, got {feats!r}")
    names = feature_names(cfg)
    if len(names) != len(set(names)):
        raise ValueError(f"features has duplicate entries: {names}")
    for f in feats:
        bad = set(f) - {'type', *_ENTRY_KEYS}
        if bad:
            raise ValueError(f"features entry {f['type']!r}: unknown keys {sorted(bad)}; valid: {sorted(_ENTRY_KEYS)}")
    needs_np = False
    for name in names:
        e = _entry_cfg(cfg, name)
        k, etp = e['spatial_k'], e['time_pool']
        if k not in (None, 0) and not (isinstance(k, int) and k > 0):
            raise ValueError(f"spatial_k ('{name}') must be a positive int, or null/0 for no spatial mixing, got {k!r}")
        if etp not in ('flat', 'learned', 'window', 'none'):
            raise ValueError(f"features['{name}'] time_pool must be flat|learned|window|none, got {etp!r}")
        ENTRY_TYPES[name].check(e)
        if e['spatial_per_atom'] and (ENTRY_TYPES[name].source != 'atom' or not k):
            raise ValueError(f"spatial_per_atom ('{name}') needs a atom-code entry and spatial_k > 0")
        if etp == 'window' and not e['window']:
            raise ValueError(f"features['{name}'] time_pool='window' requires a window=[lo, hi]")
        if etp == 'learned' and int(e['time_rank']) < 1:
            raise ValueError(f"features['{name}'] time_pool='learned' requires time_rank >= 1")
        needs_np |= ENTRY_TYPES[name].needs_patches(e)
    if needs_np and cfg.get('num_patches') is None:
        raise ValueError("this configuration needs num_patches (trial length in patches)")
    return cfg


def _entry_cfg(cfg, name):
    """One entry's effective _ENTRY_KEYS (top-level default, the entry's own value wins) plus the
    shape keys every entry needs."""
    eff = {k: cfg.get(k, _HEAD_DEFAULTS[k]) for k in _ENTRY_KEYS}    # saved heads predate newer keys
    eff.update({k: v for k, v in next(f for f in cfg['features'] if f['type'] == name).items() if k != 'type'})
    for k in ('num_patches', 'num_atoms', 'sample_freq', 'patch_stride', 'patch_len', 'num_channels', 'latent_dim'):
        eff[k] = cfg.get(k)
    return eff


def needs_atom(cfg):
    """Whether any entry in cfg['features'] needs the frozen backbone (Qlet codes, or z)."""
    return any(ENTRY_TYPES[f['type']].source in ('atom', 'latent') for f in cfg.get('features', _HEAD_DEFAULTS['features']))


def needs_raw(cfg):
    """Whether any entry in cfg['features'] needs the compiled raw signal."""
    return any(ENTRY_TYPES[f['type']].source == 'raw' for f in cfg.get('features', _HEAD_DEFAULTS['features']))


def needs_latent(cfg):
    """Whether any entry in cfg['features'] reads the encoder output z (cached with the qlets)."""
    return any(ENTRY_TYPES[f['type']].source == 'latent' for f in cfg.get('features', _HEAD_DEFAULTS['features']))


def spatial_width(cfg, name):
    """K of one entry's spatial filter: spatial_k filters, or every real channel kept separate
    (spatial_k None/0 = no mixing)."""
    return _entry_cfg(cfg, name)['spatial_k'] or cfg['num_channels']


def _entry_dim(cfg, name):
    """Feature-vector width contributed by one features[] entry."""
    return ENTRY_TYPES[name].dim(_entry_cfg(cfg, name), spatial_width(cfg, name))


def feature_dim(cfg):
    """Width of the concatenated feature vector entering the readout."""
    return sum(_entry_dim(cfg, name) for name in feature_names(cfg))
