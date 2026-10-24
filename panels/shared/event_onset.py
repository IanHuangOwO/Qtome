"""Event onset of a dataset, from its own metadata.json: what the analysis panels mark with a vertical line on a time axis."""
import json
import os


def lookup_event_onset_sample(config: dict, ds_name: str):
    """This dataset's own datas/<ds_name>/metadata.json data_metadata.event_onset_sample
    (an int sample count, or None if absent) -- a per-dataset property of which loader
    pre-event shift was applied, not a per-run setting, so it lives in the dataset's own
    metadata rather than being copy-pasted into every config that references it (see
    that file's sibling event_onset_sample_note for the per-dataset rationale). Finds
    ds_name's dataset_path by searching every dataset_params.<mode> entry (pretrain,
    finetune, ...) for a matching key. Returns None if the dataset isn't in config at
    all, or its metadata.json has no event_onset_sample -- `is not None`, not truthiness,
    at every call site: an onset of literal 0 is a real, legitimate value, not "not
    configured"."""
    dataset_path = None
    for mode_params in config.get('dataset_params', {}).values():
        if ds_name in mode_params:
            dataset_path = mode_params[ds_name].get('dataset_path')
            break
    if dataset_path is None:
        return None
    meta_path = os.path.join(dataset_path, 'metadata.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    dm = meta.get('data_metadata', {})
    if dm.get('event_onset_seconds') is not None:
        # MOABB-backed datasets store seconds (rate-independent), see IO/loader.py's
        # write_moabb_metadata; convert with this run's sample rate.
        fs = config.get('preprocess_params', {}).get('sample_freq', 200)
        return int(round(dm['event_onset_seconds'] * fs))
    return dm.get('event_onset_sample')


def event_onset_patch(config: dict, ds_name: str):
    """Event onset on a patch-index x-axis (patch n covers samples
    [n*stride, n*stride+patch_len), plotted at its centre): (onset - patch_len/2) / stride,
    or None when the dataset has no event (continuous windows)."""
    onset = lookup_event_onset_sample(config, ds_name)
    if onset is None:
        return None
    pp = config.get('preprocess_params', {})
    L = pp.get('patch_length', 50)
    stride = pp.get('patch_stride', L)
    return (onset - L / 2) / stride
