"""Channels and electrode positions: MNE's template positions (10-20, then 10-05), the polar-metadata fallback, the montages in
configs/montages.json, the channel list a run uses, and the inverse-distance interpolation of missing sites. A dataset's own
recorded positions are never used (docs/adr/0023)."""
import json
import os
from typing import Dict, List, Optional

import numpy as np


def get_standard_coords(ch_name: str) -> Optional[np.ndarray]:
    try:
        import mne
        if not hasattr(get_standard_coords, '_positions'):
            montage = mne.channels.make_standard_montage('standard_1020')
            get_standard_coords._positions = montage.get_positions()['ch_pos']
        positions = get_standard_coords._positions
        keys = {k.upper(): k for k in positions.keys()}
        if ch_name.upper() in keys:
            return positions[keys[ch_name.upper()]]
    except Exception:
        pass
    return None


# Channels excluded by default when channels_to_use is "all".
# Set include_non_eeg_channels: true in dataset_params to override.
NON_EEG_CHANNELS = {
    'EOG', 'VEOG', 'HEOG', 'EOG1', 'EOG2',
    'EMG', 'EMG1', 'EMG2',
    'ECG', 'EKG',
    'A1', 'A2', 'M1', 'M2',
    'REF', 'LREF', 'RREF',
    'STI', 'STIM', 'STATUS', 'TRIGGER',
}


HEAD_RADIUS_M = 0.095   # polar metadata coordinates are projected onto a sphere of this radius (native layout)


def channel_xyz(ch_info: Dict) -> Optional[np.ndarray]:
    """Template coordinate of one metadata channel: MNE's standard_1020 position for the label, else
    standard_1005, else the polar topomap coordinates projected onto a head sphere (radius 0.5 = the equator), else
    None. A dataset's own recorded positions are never used (tested and rejected, docs/adr/0023)."""
    p = get_standard_coords(ch_info.get('label', ''))
    if p is None:
        p = _standard_1005().get(ch_info.get('label', '').strip().lower())   # 10-05 names (FFC1h, TPP9h, AFp3h, ...)
    if p is not None:
        return np.asarray(p, dtype=np.float64)
    c = ch_info.get('coordinates')
    if c:
        th, el = np.deg2rad(c.get('polar_angle_deg', 0.0)), np.pi * c.get('polar_radius', 0.0)
        return HEAD_RADIUS_M * np.array([np.sin(el) * np.sin(th), np.sin(el) * np.cos(th), np.cos(el)])
    return None


_STD1005 = None


def _standard_1005() -> Dict[str, np.ndarray]:
    """MNE's standard_1005 positions by lower-case label (standard_1020 lacks the 10-05 half-step sites)."""
    global _STD1005
    if _STD1005 is None:
        import mne
        _STD1005 = {k.lower(): v for k, v in mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos'].items()}
    return _STD1005


IDW_K = 4   # > Nc channels: a missing canonical site = inverse-distance-squared mean of its k nearest good electrodes


def idw_matrix(pos_from: np.ndarray, pos_to: np.ndarray, k: int = IDW_K) -> np.ndarray:
    """Inverse-distance-weighted (1/d^2) interpolation from the k nearest electrodes -> [len(pos_to), len(pos_from)].
    Chosen over spherical splines (measured: on a 126-channel montage IDW had
    the lower mean and median error, and splines blew up on a noisy electrode)."""
    W = np.zeros((len(pos_to), len(pos_from)))
    for r, p in enumerate(pos_to):
        d2 = ((pos_from - p) ** 2).sum(1)
        nn = np.argsort(d2)[:k]
        w = 1.0 / np.maximum(d2[nn], 1e-12)
        W[r, nn] = w / w.sum()
    return W


def load_montage_channels(name: str) -> List[str]:
    """Looks up a named montage's ordered channel-label list from configs/montages.json —
    see docs/agents/adding-a-montage.md. Coordinates in that file are reference/QA data
    only (per-dataset coords are still resolved independently in IO/loader.py); only the
    label order matters here."""
    montage_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'montages.json')
    with open(montage_path, 'r', encoding='utf-8') as f:
        montages = json.load(f)
    if name not in montages:
        raise ValueError(f"Unknown montage '{name}' — not found in {montage_path}. "
                          f"Available: {list(montages.keys())}")
    return [ch['label'] for ch in montages[name]['channels']]


def resolve_canonical_channels(canonical_channels) -> List[str]:
    """canonical_channels: a montage name (str, looked up via load_montage_channels) or
    an already-literal channel-label list -> list. Centralizes the isinstance check every
    caller of preprocess_params.canonical_channels needs — taking len() of the raw string
    instead silently returns the name's character count (e.g. len("10-10") == 5), not an
    error."""
    if isinstance(canonical_channels, str):
        return load_montage_channels(canonical_channels)
    return canonical_channels


def resolve_target_channels(dataset_params: Dict, pp: Dict = None) -> List[str]:
    """
    Determines the unified channel list.
    If preprocess_params contains 'canonical_channels', that fixed ordered list is used
    directly — channel index = electrode identity across all datasets. It may be given as
    a literal list (custom, used as-is) or a string naming a montage in
    configs/montages.json (see docs/agents/adding-a-montage.md).
    Otherwise falls back to reading from the first dataset's metadata.
    """
    if pp:
        canonical = pp.get('canonical_channels', [])
        if canonical:
            return resolve_canonical_channels(canonical)

    first_ds_key = next(iter(dataset_params))
    first_ds_args = dataset_params[first_ds_key]
    channels_to_use = first_ds_args.get('channels_to_use', ['all'])
    include_non_eeg = first_ds_args.get('include_non_eeg_channels', False)

    if channels_to_use not in ('all', ['all']):
        return channels_to_use

    meta_path = os.path.join(first_ds_args['dataset_path'], 'metadata.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        channel_dict = json.load(f).get('data_metadata', {}).get('channels', {})

    sorted_keys = sorted(
        [k for k in channel_dict if isinstance(k, str) and k.isdigit()],
        key=lambda k: int(k)
    )
    target_channels = []
    for k in sorted_keys:
        ch_info = channel_dict[k]
        if isinstance(ch_info, dict) and 'label' in ch_info:
            label = ch_info['label']
            if include_non_eeg or label.upper() not in NON_EEG_CHANNELS:
                target_channels.append(label)
    return target_channels
