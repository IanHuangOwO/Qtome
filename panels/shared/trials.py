"""Trial and subject selection shared by the panels that draw real trials (snapshot, atom_distribution, codebook)."""
import json
import os
import random

import torch


def pick_trial(dataset, subject_id, trial: int = None, dataset_name: str = None):
    """
    Return (trial_idx, subject_id). Picks randomly if trial is None.
    dataset_name: disambiguates subject_id across multi-dataset training runs (see
    build_dataset_from_config / dataset_params.pretrain having multiple entries) —
    subject IDs are per-source-dataset, not globally unique (e.g. subject "1" can exist
    in both BETA_3s and BCICIV1_Train), so without this a collision silently picks
    whichever dataset's matching subject happened first in the concatenated tensor.
    """
    sub_data = dataset.base_dataset.subject_data
    try:
        sid = type(sub_data[0].item())(subject_id)
        indices = (sub_data == sid).nonzero(as_tuple=True)[0]
        if dataset_name is not None and indices.numel() > 0:
            names = dataset.base_dataset.dataset_names
            keep = torch.tensor([names[i] == dataset_name for i in indices.tolist()], dtype=torch.bool)
            indices = indices[keep]
    except Exception:
        indices = torch.arange(len(dataset))

    if len(indices) == 0:
        raise ValueError(f"No trials found for subject {subject_id!r}" + (f", dataset {dataset_name!r}" if dataset_name else ""))

    idx = indices[trial % len(indices)].item() if trial is not None else random.choice(indices).item()
    actual_subject = dataset.base_dataset.subject_data[idx].item()
    return idx, actual_subject


def cap_subjects(config: dict, ds_args: dict, max_trials: int, rng, min_subjects: int = 20, with_total: bool = False):
    """A shuffled subset of ds_args's subjects covering ~max_trials trials (and at least min_subjects
    subjects, subject diversity dominates EEG variance), counted from each subject's cached labels
    array without loading its data. subject_to_use ["all"] on a large dataset would otherwise load
    every trial (this once ran out of memory at ~55 GB). with_total: also return those subjects' trial count."""
    import numpy as np
    from IO.preprocessing import cache_suffix
    path = ds_args['dataset_path']
    with open(os.path.join(path, 'metadata.json'), 'r', encoding='utf-8') as f:
        subjects = sorted(json.load(f).get('data_structure', {}))
    requested = ds_args.get('subject_to_use', ['all'])
    if requested not in (['all'], 'all'):
        subjects = [s for s in subjects if s in {str(r) for r in requested}]
    rng.shuffle(subjects)
    pp = config['preprocess_params']
    # pretrain caches are event-free (continuous_seconds in the suffix), finetune caches are not: take the one on disk
    suffixes = [cache_suffix(pp['sample_freq'], pp['bandpass_filter'], pp.get('pre_event_seconds', 0.0),
                             pp.get('post_event_seconds', 0.0), c) for c in (pp.get('continuous_seconds', 0.0), 0.0)]
    picked, total = [], 0
    for sid in subjects:
        cache = next((c for c in (os.path.join(path, 'cache', f"{sid}_{x}.npz") for x in suffixes) if os.path.exists(c)), None)
        if cache is None:
            continue
        picked.append(sid)
        total += len(np.load(cache)['labels'])
        if total >= max_trials and len(picked) >= min_subjects:
            break
    return (picked, total) if with_total else picked
