"""noise_classify: does removing atoms (or the task filter) lower the classification score? Per recipe, accuracy of a shrinkage LDA on the reconstructed signal of
labelled held-out datasets (motor imagery: balanced accuracy; P300: ROC AUC), within-subject and leave-one-subject-out, mean +- SE over the seeds of each recipe
-> noise_classify.png and noise_classify.md. Dotted line: the raw signal. Needs the pretrain panel task_filter_atom_classify on each backbone."""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz
from panels.pretrain.task_filter_artifacts import ATOM_COLORS
from panels.shared.groups import bar_legend, mean_se, numbers_path, recipes

PRESETS = {'standard'}
HATCH = ('', '//', '..', 'xx')
MCOL = {'raw': '#222222', 'backbone64': '#555555', 'multi-task 0.5': '#eb6834', 'multi-task 0.1': '#b3410f', **{m: c for m, (c, _) in ATOM_COLORS.items()}}
SHOW = ('raw', 'backbone64', 'multi-task 0.5', 'multi-task 0.1', 'atoms -eye', 'atoms -muscle', 'atoms -line', 'atoms -artifacts', 'atoms -brain', 'atoms brain only')    # the methods of noise_removal


def load(path):
    """atom_classify.npz merged with its atom_classify_freq.npz (an earlier test of the frequency groups, a target of its own) when both exist."""
    out = {}
    for f in (path.replace('.npz', '_freq.npz'), path):
        if os.path.exists(f):
            z = load_npz(f); targets = [str(t) for t in z['targets']]
            out.update({k: z[k] for k in z if k != 'targets'}); out['targets'] = np.array(list(dict.fromkeys([*(str(t) for t in out.get('targets', [])), *targets])))
    return out


def compute(ctx):
    data = {}
    for r, members in recipes(ctx.groups).items():
        for g, bb in members:
            p = numbers_path(bb, 'filter', 'multi', 'atom_classify.npz')
            if os.path.exists(p) or os.path.exists(p.replace('.npz', '_freq.npz')):
                data.setdefault(r, []).append(load(p))
            else:
                print(f'  {g} ({bb}): no atom_classify numbers; left out')
    if not data:
        print('  no atom_classify numbers: run the pretrain panel task_filter_atom_classify on each backbone'); return
    targets = [t for t in map(str, next(iter(data.values()))[0]['targets']) if all(f'{t}__methods' in s for v in data.values() for s in v)]; out = {'targets': np.array(targets)}; lines = []
    for ds in targets:
        methods = [m for m in SHOW if all(m in [str(x) for x in s[f'{ds}__methods']] for v in data.values() for s in v)]
        out[f'{ds}__methods'] = np.array(methods); lines.append(f'\n{ds}: within-subject | leave-one-subject-out (mean over seeds)')
        for r, v in data.items():
            idx = [[[str(x) for x in s[f'{ds}__methods']].index(m) for m in methods] for s in v]
            for key in ('within', 'loso'):
                out[f'{r}__{ds}__{key}'] = np.array([s[f'{ds}__{key}'][i].mean(1) for s, i in zip(v, idx)])
            lines.append(f'  {r} (n={len(v)} seeds)')
            for j, m in enumerate(methods):
                lines.append(f'    {m:18s} {out[f"{r}__{ds}__within"][:, j].mean():.3f} | {out[f"{r}__{ds}__loso"][:, j].mean():.3f}')
    out['recipes'] = np.array(list(data))
    np.savez(ctx.numbers('noise_classify.npz'), **out)
    open(ctx.numbers('noise_classify.md'), 'w').write('```\n' + '\n'.join(lines) + '\n```\n'); print('\n'.join(lines))


def draw(ctx):
    try:
        z = np.load(ctx.numbers('noise_classify.npz'))
    except FileNotFoundError:
        return
    targets, recs = [str(t) for t in z['targets']], [str(r) for r in z['recipes']]
    fig, ax = plt.subplots(2, len(targets), figsize=(5.6 * len(targets), 8.5), squeeze=False)
    for j, ds in enumerate(targets):
        methods = [str(m) for m in z[f'{ds}__methods']]
        for i, key in enumerate(('within', 'loso')):
            s = {r: mean_se(z[f'{r}__{ds}__{key}']) for r in recs}; x = np.arange(len(methods)); w = 0.8 / len(recs)
            for k, r in enumerate(recs):
                ax[i, j].bar(x + (k - (len(recs) - 1) / 2) * w, s[r][0], w * 0.95, yerr=s[r][1], color=[MCOL.get(m, '#888888') for m in methods], hatch=HATCH[k % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
            ax[i, j].set_xticks(x); ax[i, j].set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
            raw = np.mean([s[r][0][methods.index('raw')] for r in recs]); ax[i, j].axhline(raw, color='k', lw=.8, ls=':'); ax[i, j].axhline(0.5, color='#bbbbbb', lw=.6)
            lo = min(m.min() for m, _ in s.values()); hi = max(m.max() for m, _ in s.values()); ax[i, j].set_ylim(max(0.3, lo - 0.1), min(1.0, hi + 0.08))
            ax[i, j].set_ylabel(('within-subject' if key == 'within' else 'leave-one-subject-out') + ' score')
        ax[0, j].set_title(ds, fontsize=10, fontweight='bold')
    if len(recs) > 1:
        ax[0, 0].legend(handles=bar_legend(recs)[:-1], fontsize=8)
    fig.suptitle(textwrap.fill('Classification after removal (dotted: raw signal; below it = the removed part carried what the decoder uses): mean ± SE over seeds', width=int(40 * len(targets))), fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = ctx.figure('noise_classify.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
