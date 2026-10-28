"""Saved numbers: what a panel's compute() writes and its draw() reads (so --redraw never recomputes), as NPZ arrays."""
import os

import numpy as np


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_npz(path):
    """Dict of arrays (loaded fully: the file handle is closed)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
