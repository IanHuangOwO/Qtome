"""noise_removal_synthetic: artificial noise (eye, muscle, electrode, 60 Hz line) added to a clean dataset at TUAR's strength, per recipe: noise left (1 = untouched) and
signal kept (1 = all), known exactly because the clean signal and the noise are known; mean +- SE over the seeds of each recipe -> noise_removal_synthetic.png and
noise_removal_synthetic.md. Needs the pretrain panel task_filter_noise_removal_synthetic on each backbone (--syn-dataset names the dataset, default PhysionetMI)."""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz
from panels.pretrain.task_filter_artifacts import ATOM_COLORS
from panels.shared.groups import bar_legend, color_of, describe, db, grouped_bars, load_all, mean_se

PRESETS = {'standard'}
MARKERS = ('o', 's', '^', 'D')
HATCH = ('', '//', '..', 'xx')
SYN = 'a clean dataset'
SHOW = ('backbone64', 'multi-task 0.5', 'multi-task 0.1', 'atoms -eye', 'atoms -muscle', 'atoms -line', 'atoms -artifacts', 'atoms brain only', 'ASR', 'ICLabel', 'ASR atoms 1.5', 'ASR atoms 1.5 + atoms -line')
KINDS = ('eye', 'muscle', 'electrode', 'line')
WHAT = {'eye': 'frontal 1-10 Hz power', 'muscle': '30-90 Hz power', 'electrode': 'peak amplitude²', 'line': '59-61 Hz power'}


def add_args(flags):
    flags.add('--noise-dataset', default='PhysionetMI', help='noise_removal_synthetic (group): the clean dataset of the pretrain panel task_filter_noise_removal_synthetic')


def load(path):
    """synthetic_<ds>.npz plus the rows of synthetic_<ds>_asr.npz (a partial run of the atom-space ASR methods) that it does not have."""
    z = load_npz(path); out = {k: z[k] for k in z}
    extra = path.replace('.npz', '_asr.npz')
    if os.path.exists(extra):
        e = load_npz(extra); em = [str(m) for m in e['methods']]; have = [str(m) for m in out['methods']]; rows = [i for i, m in enumerate(em) if m not in have]
        if rows:
            for k in out:
                if k not in ('methods', 'levels') and k in e and out[k].ndim == 2:
                    n = max(out[k].shape[1], e[k].shape[1]); pad = lambda a: np.pad(a, ((0, 0), (0, n - a.shape[1])), constant_values=np.nan)
                    out[k] = np.concatenate([pad(out[k]), pad(e[k][rows])])
            out['methods'] = np.array([*have, *(em[i] for i in rows)])
    return out


def compute(ctx):
    ds = ctx.args.noise_dataset
    data = load_all(ctx.groups, load, 'filter', 'multi', f'synthetic_{ds}.npz')
    if not data:
        print('  no synthetic numbers: run the pretrain panel task_filter_noise_removal_synthetic on each backbone'); return
    levels = [float(v) for v in next(iter(data.values()))[0]['levels']]; mid = levels[len(levels) // 2]
    methods = [m for m in SHOW if all(m in [str(x) for x in s['methods']] for v in data.values() for s in v)]
    out = {'methods': np.array(methods), 'level': np.array(mid)}; lines = [f'{ds}, noise {mid:g} dB over the clean window: noise left | signal kept (mean over seeds)']
    for r, v in data.items():
        idx = [[[str(x) for x in s['methods']].index(m) for m in methods] for s in v]
        for kind in KINDS:
            for q in ('leak', 'signal', 'artifact', 'control'):
                out[f'{r}__{kind}__{q}'] = np.array([np.nanmean(s[f'{kind}__{mid:g}__{q}'][i], axis=1) for s, i in zip(v, idx)])
        lines.append(f'  {r} (n={len(v)} seeds)')
        for kind in KINDS:
            lines.append(f'    {kind}: ' + ', '.join(f'{m} {out[f"{r}__{kind}__leak"][:, j].mean():.2f}|{out[f"{r}__{kind}__signal"][:, j].mean():.2f}' for j, m in enumerate(methods)))
    np.savez(ctx.numbers('noise_removal_synthetic.npz'), recipes=np.array(list(data)), **out)
    open(ctx.numbers('noise_removal_synthetic.md'), 'w').write('```\n' + '\n'.join(lines) + '\n```\n'); print('\n'.join(lines))


def draw(ctx):
    try:
        z = np.load(ctx.numbers('noise_removal_synthetic.npz'))
    except FileNotFoundError:
        return
    methods, recs = [str(m) for m in z['methods']], [str(r) for r in z['recipes']]; col = color_of(recs); n = len(recs); x = np.arange(len(methods)); w = 0.8 / n
    mcol = {'backbone64': '#555555', 'ASR': '#1baf7a', 'ICLabel': '#eda100', 'ASR atoms 1.5': '#e377c2', 'ASR atoms 1.5 + atoms -line': '#d62728', 'multi-task 0.5': '#eb6834', 'multi-task 0.1': '#b3410f', **{m: c for m, (c, _) in ATOM_COLORS.items()}}
    fig, ax = plt.subplots(3, len(KINDS), figsize=(3.9 * len(KINDS), 13.5), squeeze=False)
    for j, kind in enumerate(KINDS):
        a = ax[0, j]                                                          # row 1 as in noise_removal: removal against damage; the damage is the exact signal kept
        for i, r in enumerate(recs):
            for m in methods:
                a.scatter(db(z[f'{r}__{kind}__artifact'])[:, methods.index(m)].mean(), z[f'{r}__{kind}__signal'][:, methods.index(m)].mean(), color=mcol.get(m, '#888888'), marker=MARKERS[i % len(MARKERS)], s=60, zorder=3)
        a.scatter([0], [1], color='#555555', marker='s', s=60, zorder=3); a.axvline(0, color='k', lw=.5, ls=':')
        a.set_xlabel('noise power change (dB)'); a.set_ylabel('signal kept (1 = all, exact)' if j == 0 else ''); a.set_title(f'{kind}\n{WHAT[kind]}', fontsize=9, fontweight='bold')
        if j == 0:
            h = [plt.Line2D([], [], marker='s', color='#555555', ls='', label='input')] + [plt.Line2D([], [], marker='o', color=c, ls='', label=m) for m, c in mcol.items() if m in methods]
            h += [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], color='k', mfc='none', ls='', label=f'recipe {r}') for i, r in enumerate(recs)]
            a.legend(handles=h, fontsize=7, loc='lower left')
        b = ax[1, j]                                                          # row 2: noisy windows (method colour; hatch = recipe) next to the control on clean windows (grey)
        for i, r in enumerate(recs):
            off = (i - (n - 1) / 2) * w
            for kk, c, d in (('artifact', [mcol.get(m, '#888888') for m in methods], -w / 4), ('control', '#cccccc', w / 4)):
                mu, se = mean_se(db(z[f'{r}__{kind}__{kk}']))
                b.bar(x + off + d, mu, w / 2 * 0.95, yerr=se, color=c, hatch=HATCH[i % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
        b.axhline(0, color='k', lw=.6); b.set_ylabel('power change (dB; 0 = unchanged)' if j == 0 else ''); b.set_xticks(x); b.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
        if j == 0:
            b.legend(handles=bar_legend(recs), fontsize=7, loc='lower left')
        c = ax[2, j]                                                          # row 3: signal-to-noise change, exact: 20 log10(signal kept / noise left); 0 = as noisy as the input
        for i, r in enumerate(recs):
            snr = 20 * np.log10(np.maximum(z[f'{r}__{kind}__signal'], 1e-2) / np.maximum(z[f'{r}__{kind}__leak'], 1e-2)); mu, se = mean_se(snr)
            c.bar(x + (i - (n - 1) / 2) * w, mu, w * 0.95, yerr=se, color=[mcol.get(m, '#888888') for m in methods], hatch=HATCH[i % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
        c.axhline(0, color='k', lw=.6); c.set_ylabel('SNR change (dB; > 0 = cleaner)' if j == 0 else ''); c.set_xticks(x); c.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    fig.suptitle(textwrap.fill(f'Artificial noise at {float(z["level"]):g} dB over the clean window (TUAR\'s median strength) on {SYN}: same layout as noise_removal_TUAR, row 3 the exact signal-to-noise change 20 log10(signal kept / noise left) (mean over seeds, ± SE on the bars)', width=int(24 * len(KINDS))), fontweight='bold')
    fig.tight_layout(rect=describe(fig, methods, 0.11)); out = ctx.figure('noise_removal_synthetic.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
