"""Saved numbers: what a panel's compute() writes and its draw() reads (so --redraw never recomputes). JSON for small
tables, NPZ for arrays; both are written by these helpers so the convention is one."""
import json
import os

import numpy as np


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o).__name__}')


def save_json(path, obj, indent=2):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent, default=_plain)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_npz(path):
    """Dict of arrays (loaded fully: the file handle is closed)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
