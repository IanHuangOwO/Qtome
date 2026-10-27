"""task_filter_atom_classify: does removing atoms (or the task filter) lower the classification score? Atom removal drops the atoms whose top ICLabel
class is in a group (panels/pretrain/task_filter_artifacts.ATOM_GROUPS; needs the atom_labels panel first), the multi-task filter keeps what the heads use.
On labelled held-out datasets (--cls-targets, default PhysionetMI left / right fist, BNCI2014004 left / right hand: motor imagery; BNCI2014008: P300)
every signal is the backbone's reconstruction of the imputed 64-channel encoding, with the real channels read out (as task_filter_eval). Scores, per subject then
mean +- SE over subjects: 'within' = 5-fold shrinkage LDA inside the subject (balanced accuracy; P300: ROC AUC), 'loso' = leave-one-subject-out (task_filter/tests.py).
Features: MI log 8-13 and 13-30 Hz power per channel; P300 10-sample means per channel. A method that lowers the score removed something the task uses;
one that leaves it level removed only what the task does not need (the brain atoms it dropped carried nothing for this decoder). SSVEP is not tested here.
-> pretrain/analysis/filter/<source>/atom_classify.npz, (a figure only with --redraw; the group panel noise_classify draws the all-seed one)"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from panels.core.numbers import load_npz, save_npz
from panels.pretrain.task_filter_artifacts import ATOM_COLORS, ATOM_GROUPS
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.tests import _setup, lda, lda_auc, loso, mi_features, p300_features, subjects

PRESETS = set()
PARADIGM = {'PhysionetMI': 'mi', 'BNCI2014004': 'mi', 'BNCI2014001': 'mi', 'BNCI2014008': 'p300', 'BNCI2014009': 'p300'}
CLASSES = {'PhysionetMI': (1, 2)}                                         # label codes kept (left, right fist); other datasets use every window


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--cls-targets', nargs='+', default=['PhysionetMI', 'BNCI2014004', 'BNCI2014008'], choices=sorted(PARADIGM), help='task_filter_atom_classify: labelled datasets (BNCI2014001 / 009 are filter sources: the multi-task rows are then optimistic)')
    flags.add('--cls-methods', nargs='+', help='task_filter_atom_classify: only these methods (default all), e.g. raw backbone64 "atoms -brain" "atoms -line"')
    flags.add('--cls-tag', default='', help='task_filter_atom_classify: suffix of the output files, to keep a small test apart from the full run')
    flags.add('--cls-subjects', type=int, default=20, help='task_filter_atom_classify: subjects per dataset (spread out)')


def compute(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    atom_keep = {}
    if os.path.exists(ctx.numbers('atom_labels.npz')):
        zl = load_npz(ctx.numbers('atom_labels.npz')); top = [str(zl['classes'][i]) for i in zl['probs'].argmax(1)]
        atom_keep = {n: torch.tensor([t not in g for t in top], dtype=torch.float32) for n, g in ATOM_GROUPS.items()}
        print('  atoms dropped: ' + ', '.join(f'{n} {int((1 - m).sum())}/{len(m)}' for n, m in atom_keep.items()))
    else:
        print('  atom removal skipped: run the atom_labels panel first')
    arrays = {}
    for dst in ctx.args.cls_targets:
        p300_ = PARADIGM[dst] == 'p300'; dst_r, dst_c, sel, encode, ops = _setup(tf, dst); dec = tf.backbone.decode
        files = subjects(dst); step = max(1, len(files) // ctx.args.cls_subjects); files = files[::step][:ctx.args.cls_subjects]
        rows, ys, feats = {}, [], {}
        for f in files:
            z = np.load(f); y = z['labels']; T = int(z['valid_end'][0]); x = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
            if dst in CLASSES:
                keep = np.isin(y, CLASSES[dst]); x, y = x[keep], (y[keep] == CLASSES[dst][1]).astype(int)
            if len(np.unique(y)) < 2 or np.bincount(y).min() < 5:
                continue
            x = x / x.std((1, 2), keepdims=True); g = encode(x, True)
            sig = {'raw': x, 'backbone64': dec(g[:, :, sel])[..., :T]}
            for q, o in ops.items():
                if any(abs(q - a) < 1e-9 for a in (0.5, 0.1)):
                    sig[f'multi-task {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', o, g))[..., :T]
            for n, m in atom_keep.items():
                sig[n] = dec(g[:, :, sel] * m.to(g.device, g.dtype).view(1, 1, 1, -1, 1))[..., :T]
            if ctx.args.cls_methods:
                sig = {m: w for m, w in sig.items() if m in ctx.args.cls_methods}
            ys.append(y)
            for m, w in sig.items():
                rows.setdefault(m, []).append(float((lda_auc if p300_ else lda)(w, y)))
                feats.setdefault(m, []).append((p300_features if p300_ else mi_features)(w))
            print(f'  {dst} subject {os.path.basename(f).split("_")[0]}: {len(y)} trials', flush=True)
        methods = list(rows)
        arrays[f'{dst}__methods'] = np.array(methods)
        arrays[f'{dst}__within'] = np.array([rows[m] for m in methods])
        arrays[f'{dst}__loso'] = np.array([loso(feats[m], ys, p300_) for m in methods])
        metric = 'ROC AUC' if p300_ else 'balanced accuracy'
        print(f'\n  {dst} ({len(ys)} subjects, {metric}): within-subject | leave-one-subject-out, mean +- SE over subjects')
        for i, m in enumerate(methods):
            w_, l_ = arrays[f'{dst}__within'][i], arrays[f'{dst}__loso'][i]
            print(f'    {m:18s} {w_.mean():.3f} +-{w_.std(ddof=1) / np.sqrt(len(w_)):.3f} | {l_.mean():.3f} +-{l_.std(ddof=1) / np.sqrt(len(l_)):.3f}')
    save_npz(ctx.numbers('filter', ctx.args.filter_source, f'atom_classify{ctx.args.cls_tag}.npz'), targets=np.array(ctx.args.cls_targets), **arrays)


def draw(ctx):
    if not ctx.redraw:                                                      # numbers only: the group panels draw the figures; a single backbone's on request (--redraw)
        return
    path = ctx.numbers('filter', ctx.args.filter_source, f'atom_classify{ctx.args.cls_tag}.npz')
    if not os.path.exists(path):
        return
    z = load_npz(path); targets = [str(t) for t in z['targets']]
    colors = {'raw': '#000000', 'backbone64': '#555555', 'multi-task 0.5': '#eb6834', 'multi-task 0.1': '#6d1d05', **{m: c for m, (c, _) in ATOM_COLORS.items()}}
    fig, ax = plt.subplots(2, len(targets), figsize=(5.2 * len(targets), 8.5), squeeze=False)
    for j, dst in enumerate(targets):
        methods = [str(m) for m in z[f'{dst}__methods']]
        for i, key in enumerate(('within', 'loso')):
            a = z[f'{dst}__{key}']; mu = a.mean(1); se = a.std(1, ddof=1) / np.sqrt(a.shape[1]); x = np.arange(len(methods))
            ax[i, j].bar(x, mu, yerr=se, color=[colors.get(m, '#999999') for m in methods], capsize=2)
            ax[i, j].axhline(mu[methods.index('raw')], color='k', lw=.8, ls=':', label='raw'); ax[i, j].axhline(0.5, color='#bbbbbb', lw=.6)
            ax[i, j].set_xticks(x); ax[i, j].set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
            ax[i, j].set_ylim(max(0.3, mu.min() - 0.1), min(1.0, mu.max() + 0.08))
            ax[i, j].set_ylabel(('within-subject' if key == 'within' else 'leave-one-subject-out') + (' ROC AUC' if PARADIGM[dst] == 'p300' else ' balanced accuracy'))
            ax[i, j].set_title(f'{dst} ({"P300" if PARADIGM[dst] == "p300" else "motor imagery"}, {a.shape[1]} subjects)' if i == 0 else '', fontsize=9, fontweight='bold')
    fig.suptitle('Classification after removal: dotted line = raw signal. A bar below it means the removed atoms carried what the decoder uses.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ctx.figure_in('filter', ctx.args.filter_source, f'atom_classify{ctx.args.cls_tag}.png'); fig.savefig(out, dpi=110); plt.close(fig)
    print(f'  -> {out}')
