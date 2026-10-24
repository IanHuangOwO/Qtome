"""Where a task filter finds its inputs: the finetune cells, the feature-cache gains, the trained heads, the channel names and the layout of a dataset."""
import glob
import json
import os

import numpy as np
import torch

from IO.dataset import build_dataset_from_config
from IO.channels import get_standard_coords
from model.Qtome.head.feature_head import FeatureHead

MONT = json.load(open('configs/montages.json'))['10-10']['channels']
CHANNELS = [m['label'] for m in MONT]                                     # the 64 slots, in slot order
FS = 200                                                                  # sample rate


def cell_dir(bb, ds):                                                     # bb = output/Qtome/<backbone>
    return f'{bb}/finetune/allatom/{ds}_impute64_loso_seed1'


def gains(bb, ds, C, N):
    """{subject: (amp [T, N, C, S, 2], labels)} from the newest feature cache folder with C channels (and N patches)."""
    for d in sorted(glob.glob(f'{bb}/pretrain/feature_cache/{ds}/*'), key=os.path.getmtime, reverse=True):
        fs = sorted(glob.glob(d + '/*.npz'))
        if np.load(fs[0])['amp'].shape[1:3] in ((N, C), (np.load(fs[0])['amp'].shape[1], C) if N is None else ()):   # N None: any patch count
            return {os.path.basename(f)[:-4]: (lambda z: (torch.from_numpy(z['amp'].astype(np.float32)), z['labels']))(np.load(f)) for f in fs}
    raise FileNotFoundError(f'no feature cache of {ds} with N {N}, C {C} under {bb}: run a finetune cell of it first')


def names(ds):
    """A dataset's channel labels in recording order and in gain order (slot order)."""
    lab = [v['label'] for v in json.load(open(f'datas/finetune/{ds}/metadata.json'))['data_metadata']['channels'].values() if isinstance(v, dict)]
    free = iter(i for i, m in enumerate(CHANNELS) if m not in lab)        # as IO/dataset.py: a channel outside
    slot = {n: CHANNELS.index(n) if n in CHANNELS else next(free) for n in lab}   # the montage takes the next free slot
    return lab, sorted(lab, key=slot.get)


def load_head(cell, fold):
    hk = torch.load(f'{cell}/finetune/run_{fold}/head.pth', map_location='cpu', weights_only=False)
    hc = hk['head_config']; head = FeatureHead(hc); head.load_state_dict(hk['model_state_dict']); head.eval()
    assert hc['num_channels'] == 64, f'{cell}: head on {hc["num_channels"]} channels, not the imputed 64'
    return hc, torch.cat([head.spatials[b].weight for b in head.entries]).detach()   # stacked filters [2K, 64]


def layout(bb, ds):
    """Valid slots, coordinates and preprocessing of a dataset, from the config of its loso cell (or, for a dataset
    without one, a P300 cell's with only the dataset swapped); missing slots at their template position if the cell
    imputes them."""
    path = f'{bb}/finetune/allatom/{ds}_loso_seed1/artifacts/config.json'
    if os.path.exists(path):
        cfg = json.load(open(path))
    else:
        cfg = json.load(open(f'{bb}/finetune/allatom/BNCI2014009_loso_seed1/artifacts/config.json'))
        cfg['dataset_params']['finetune'] = {ds: {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['1'], 'channels_to_use': ['all']}}
    return _layout(cfg, ds)


def layout64(bb, ds):                                                     # the 64 imputed slots of an imputed cell
    return _layout(json.load(open(f'{cell_dir(bb, ds)}/artifacts/config.json')), ds)


def _layout(cfg, ds):
    d = cfg['dataset_params']['finetune'][ds]
    d['subject_to_use'] = [d['subject_to_use'][0] if d['subject_to_use'] != ['all'] else '1']
    base = build_dataset_from_config(cfg, mode='finetune').base_dataset; valid, coords = base.all_valid_channels[0].clone(), base.all_coords[0].clone()
    if d.get('impute_missing_channels'):                                  # as cache_feature.py: missing slots at their standard position
        for i in torch.nonzero(~valid).flatten().tolist(): coords[i] = torch.as_tensor(get_standard_coords(base.channel_names[i]), dtype=coords.dtype)
        valid[:] = True
    return valid, coords, cfg['preprocess_params']
