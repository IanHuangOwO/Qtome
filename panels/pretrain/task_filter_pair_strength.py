"""task_filter_pair_strength: how strong are TUAR's annotated electrode events on their own bipolar pair? For every elec (and musc_elec) window of the spread-out patients, the peak
amplitude squared of each annotated pair (99.9th percentile of |a - b|, the measure of task_filter_artifacts' pair column) over the same pair's median on that patient's clean windows,
in dB; the 25th / 50th / 75th percentile over the (window, pair) items are the strengths the synthetic electrode pop of task_filter_noise_removal_synthetic is set to
(--syn-pair-levels). Windows are normalised by their own standard deviation as everywhere. Needs no backbone result, only the montage of the filter context.
-> pretrain/analysis/filter/<source>/pair_strength_TUAR.json"""
import json
import os

import numpy as np

from panels.pretrain.task_filter_artifacts import pair_index, tuar_coverage, tuar_pairs
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.tests import _setup, subjects

PRESETS = set()


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--pair-patients', type=int, default=40, help='task_filter_pair_strength: patients with elec windows to use')


def peak2(v, a, b):
    return np.percentile(np.abs(v[a] - v[b]), 99.9) ** 2


def compute(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    dst_r, dst_c, sel, encode, ops = _setup(tf, 'TUAR'); up = [n.upper() for n in dst_c]
    files = subjects('TUAR'); order = [files[i] for i in np.argsort((np.arange(len(files)) * 0.6180339887) % 1)]
    rises = {'elec': [], 'musc_elec': []}; used = 0
    for f in order:
        if used >= ctx.args.pair_patients:
            break
        z = np.load(f); T = int(z['valid_end'][0]); sub = int(os.path.basename(f).split('_')[0])
        raw = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
        cov, anyc = tuar_coverage(sub)
        if len(anyc) != len(raw) or not np.array_equal((anyc >= 0.5).astype(int), z['labels']):
            continue
        wp = tuar_pairs(sub); clean = np.flatnonzero(anyc == 0)
        if len(clean) < 8:
            continue
        x = raw / raw.std((1, 2), keepdims=True); got = False
        for k in rises:
            idx = np.flatnonzero((cov[k] >= 0.5) & (anyc - cov[k] < 0.1))
            for i in idx:
                for p in sorted(wp[i][k]):
                    pi = pair_index(p, up)
                    if pi is None:
                        continue
                    base = np.median([peak2(x[j], *pi) for j in clean[:30]])
                    rises[k].append(10 * np.log10(peak2(x[i], *pi) / base)); got = True
        used += got
    res = {k: {'n': len(v), 'percentiles_25_50_75': [float(p) for p in np.percentile(v, [25, 50, 75])] if v else None} for k, v in rises.items()}
    res['patients'] = used
    path = ctx.numbers('filter', ctx.args.filter_source, 'pair_strength_TUAR.json'); os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, 'w'), indent=1); print(json.dumps(res, indent=1))


def draw(ctx):
    pass                                                                    # numbers only
