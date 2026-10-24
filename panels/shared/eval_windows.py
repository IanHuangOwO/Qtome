"""The held-out pretrain windows and the fixed test masks that the backbone checks share (backbone_eval, atom_usage, attention_range,
long_context): every backbone sees the same windows and the same masks."""
import copy
import zlib

import torch

from IO.dataset import build_dataset_from_config, split_pretrain_subjects
from IO.masking import ChannelClusterMask, RandomChannelMask, TimeBlockMask, random_token_mask


def val_config(config):
    """The val half of train_pretrain.py's subject split (IO/dataset.py's split_pretrain_subjects)."""
    cfg = copy.deepcopy(config)
    ratio = config['training_params']['pretrain'].get('train_val_split', 0.9)
    split = split_pretrain_subjects(config['dataset_params']['pretrain'], ratio)
    cfg['dataset_params']['pretrain'] = {k: dict(v, subject_to_use=split[k][1])
                                         for k, v in config['dataset_params']['pretrain'].items() if split[k][1]}
    return cfg


def eval_windows(config, max_windows):
    """Held-out windows, fixed order. Masks/subsampling of the dataset itself are unused."""
    ds = build_dataset_from_config(val_config(config), mode='pretrain')
    idx = torch.randperm(len(ds.base_dataset), generator=torch.Generator().manual_seed(0))[:max_windows].tolist()
    return ds, idx


def batches(ds, idx, bs=32):
    for i in range(0, len(idx), bs):
        items = [ds[j] for j in idx[i:i + bs]]
        x, coords, t, valid = (torch.stack([torch.as_tensor(it[k]) for it in items]) for k in (0, 1, 3, 5))
        yield idx[i:i + bs], x.float(), coords.float(), t.long(), valid.bool()


def make_mask(kind, valid_tok, coords, window_id, names_idx, named=None):
    """valid_tok [C, N] bool -> (mask [C, N], score [C, N]); score = tokens that count. named [C] bool: slots that hold
    their canonical-name channel (EEGDataset.all_named_slots; non-canonical channels sit in free
    slots) -- the name-based motor3_to_bci22 test reads only those."""
    torch.manual_seed(1_000_003 * window_id + zlib.crc32(kind.encode()) % 997)   # stable across processes
    C, N = valid_tok.shape
    if kind == 'token_runs':
        m = random_token_mask(C, N, 0.5, valid_tok.flatten(), time_run=3).view(C, N)
    elif kind == 'random_channel':
        m = RandomChannelMask().generate(valid_tok, coords, 0.3)
    elif kind == 'channel_cluster':
        m = ChannelClusterMask().generate(valid_tok, coords, 0.3)
    elif kind == 'time_block':
        m = TimeBlockMask().generate(valid_tok, coords, 0.3)
    elif kind == 'time_block_short':
        m = TimeBlockMask(run_patches=(4, 4)).generate(valid_tok, coords, 4 / N)
    elif kind in ('time_block_2', 'time_block_3'):   # one hole of 2 / 3 patches: 0.25 / 0.5 s with no sample visible
        k = int(kind[-1])
        m = TimeBlockMask(run_patches=(k, k)).generate(valid_tok, coords, k / N)
    elif kind == 'motor3_to_bci22':
        real = valid_tok.any(1) & (named if named is not None else True)
        motor = torch.zeros(C, dtype=torch.bool); motor[names_idx['motor-3']] = True
        bci = torch.zeros(C, dtype=torch.bool); bci[names_idx['bci-22']] = True
        if int(real.sum()) < 32 or not (motor & real).sum() == 3 or int((bci & real & ~motor).sum()) < 10:
            return None, None
        m = valid_tok & ~motor[:, None]
        return m, m & (bci & ~motor)[:, None]
    return m, m
