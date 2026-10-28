"""TUAR patient windows shared by the artifact panels that score on annotated windows: a patient's cached windows with the label coverage of its annotation, and the stacking of
chosen window sets into one normalised array (moved here from task_filter_zsubspace, the removed z-subspace test)."""
import os

import numpy as np

from panels.pretrain.task_filter_artifacts import KINDS, tuar_coverage


def dst_windows(f, nw, dst_r, dst_c):
    """One TUAR patient -> (patient, raw [N, C, T] of the montage's channels, {label: window indices}) or None when the annotation layout does not match."""
    z = np.load(f); T = int(z['valid_end'][0]); sub = int(os.path.basename(f).split('_')[0])
    raw = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
    cov, anyc = tuar_coverage(sub)
    if len(anyc) != len(raw) or not np.array_equal((anyc >= 0.5).astype(int), z['labels']):
        return None
    rng = np.random.default_rng(sub)
    idx = {k: rng.permutation(np.flatnonzero((cov[k] >= 0.5) & (anyc - cov[k] < 0.1)))[:nw] for k in KINDS}
    idx['clean'] = rng.permutation(np.flatnonzero(anyc == 0))[:int(1.5 * nw)]
    return sub, raw, idx


def stack(raw, idx, sets):
    pos, allidx = {}, []
    for k in sets:
        pos[k] = np.arange(len(allidx), len(allidx) + len(idx[k])); allidx += list(idx[k])
    allidx = np.array(allidx); x = raw[allidx]
    return pos, x / x.std((1, 2), keepdims=True)
