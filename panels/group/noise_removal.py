"""noise_removal: what each removal method does to the artifacts of TUAR and TUEV, and to the clean brain signal, per recipe: mean +- SE over the seeds of
each recipe -> noise_removal_<dataset>.png and noise_removal.md. Bars: artifact power change in dB (0 = unchanged, lower = more removed) next to grey bars: the control (the
same method on clean windows: a method that removes the artifact and the control equally is a frequency filter); top row: brain kept (correlation with the
clean windows) against the artifact power change, one marker shape per recipe; bottom row: the same on a linear scale (output power / input power). Needs the pretrain panel task_filter_artifacts (TUAR and TUEV) and, for atom removal, atom_labels on each backbone."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz
from panels.pretrain.task_filter_artifacts import ATOM_COLORS, COLUMNS, TUEV_KINDS
from panels.shared.groups import bar_legend, color_of, describe, db, grouped_bars, load_all, mean_se

PRESETS = {'standard'}
MARKERS = ('o', 's', '^', 'D')
HATCH = ('', '//', '..', 'xx')
SHOW = ('backbone64', 'multi-task 0.5', 'multi-task 0.1', 'atoms -eye', 'atoms -muscle', 'atoms -line', 'atoms -artifacts', 'atoms -brain', 'atoms brain only', 'ASR', 'ICLabel')
WHAT = {'eye left': 'frontal 1-10 Hz power', 'muscle left': '30-90 Hz power', 'peak left': 'peak amplitude²', 'line left': '59-61 Hz power (real line noise)'}
NAME = {'eyem': 'eye movement', 'musc': 'muscle', 'elec': 'electrode', 'eyem_musc': 'eyem + musc', 'musc_elec': 'musc + elec', None: 'line noise'}


def load(path):
    z = load_npz(path)
    return {k: z[k] for k in z if '__' in k or k in ('methods', 'brain_kept')}


def compute(ctx):
    lines = []
    for ds in ('TUAR', 'TUEV'):
        data = load_all(ctx.groups, load, 'filter', 'multi', f'artifacts_{ds}.npz')
        if not data:
            continue
        methods = [m for m in SHOW if all(m in [str(x) for x in s['methods']] for v in data.values() for s in v)]
        cols = [(k, sc) for k, sc in COLUMNS if all(f'{k or "line"}__{sc}'.replace(' ', '_') in s for v in data.values() for s in v)]
        out = {'methods': np.array(methods), 'cols': np.array([f'{k or "line"}__{sc}'.replace(' ', '_') for k, sc in cols])}
        for r, v in data.items():
            idx = [[[str(x) for x in s['methods']].index(m) for m in methods] for s in v]
            for k, sc in cols:
                key = f'{k or "line"}__{sc}'.replace(' ', '_')
                out[f'{r}__{key}'] = np.array([db(np.nanmean(s[key][i], axis=1)) for s, i in zip(v, idx)])
                out[f'{r}__{key}__control'] = np.array([db(np.nanmean(s[key + '__control'][i], axis=1)) for s, i in zip(v, idx)])
            out[f'{r}__brain'] = np.array([np.nanmean(s['brain_kept'][i], axis=1) for s, i in zip(v, idx)])
        np.savez(ctx.numbers(f'noise_removal_{ds}.npz'), recipes=np.array(list(data)), **out)
        lines.append(f'\n{ds}: artifact power change in dB (control, clean windows) | brain kept; mean over seeds')
        for r, v in data.items():
            lines.append(f'  {r} (n={len(v)} seeds)')
            for j, m in enumerate(methods):
                lines.append(f'    {m:18s} ' + ' '.join(f'{out[f"{r}__{k}"][:, j].mean():6.1f} ({out[f"{r}__{k}__control"][:, j].mean():5.1f})' for k in map(str, out['cols'])) +
                             f' | {out[f"{r}__brain"][:, j].mean():.3f}')
    if not lines:
        print('  no artifacts_TUAR / artifacts_TUEV numbers: run the pretrain panel task_filter_artifacts on each backbone'); return
    open(ctx.numbers('noise_removal.md'), 'w').write('```\n' + '\n'.join(lines) + '\n```\n'); print('\n'.join(lines))


def draw(ctx):
    for ds in ('TUAR', 'TUEV'):
        try:
            z = np.load(ctx.numbers(f'noise_removal_{ds}.npz'))
        except FileNotFoundError:
            continue
        methods, cols, recs = [str(m) for m in z['methods']], [str(c) for c in z['cols']], [str(r) for r in z['recipes']]; col = color_of(recs); n = len(recs)
        mcol = {'backbone64': '#555555', 'ASR': '#1baf7a', 'ICLabel': '#eda100', 'multi-task 0.5': '#eb6834', 'multi-task 0.1': '#b3410f', **{m: c for m, (c, _) in ATOM_COLORS.items()}}
        fig, ax = plt.subplots(3, len(cols), figsize=(3.6 * len(cols), 13.5), squeeze=False); x = np.arange(len(methods)); w = 0.8 / n
        for j, key in enumerate(cols):
            k, sc = key.split('__', 1); a = ax[0, j]
            for i, r in enumerate(recs):                                      # row 1: removal against damage, one marker shape per recipe, one colour per method
                mk = MARKERS[i % len(MARKERS)]
                for m in methods:
                    a.scatter(z[f'{r}__{key}'][:, methods.index(m)].mean(), z[f'{r}__brain'][:, methods.index(m)].mean(), color=mcol.get(m, '#888888'), marker=mk, s=60, zorder=3)
            a.scatter([0], [1], color='#555555', marker='s', s=60, zorder=3); a.axvline(0, color='k', lw=.5, ls=':')
            a.set_xlabel('artifact power change (dB)'); a.set_ylabel('brain kept on clean windows (correlation)' if j == 0 else '')
            a.set_title(f'{NAME[None if k == "line" else k]}\n{WHAT[sc.replace("_", " ")]}', fontsize=9, fontweight='bold')
            if j == 0:
                h = [plt.Line2D([], [], marker='s', color='#555555', ls='', label='input')] + [plt.Line2D([], [], marker='o', color=c, ls='', label=m) for m, c in mcol.items() if m in methods]
                h += [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], color='k', mfc='none', ls='', label=f'recipe {r}') for i, r in enumerate(recs)]
                a.legend(handles=h, fontsize=7, loc='lower left')
            b = ax[1, j]                                                      # row 2: artifact windows (method colour; hatch = recipe) next to the control on clean windows (grey)
            for i, r in enumerate(recs):
                off = (i - (n - 1) / 2) * w
                for kk, c, d in ((key, [mcol.get(m, '#888888') for m in methods], -w / 4), (key + '__control', '#cccccc', w / 4)):
                    mu, se = mean_se(z[f'{r}__{kk}'])
                    b.bar(x + off + d, mu, w / 2 * 0.95, yerr=se, color=c, hatch=HATCH[i % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
            b.axhline(0, color='k', lw=.6); b.set_ylabel('power change (dB; 0 = unchanged)' if j == 0 else '')
            b.set_xticks(x); b.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
            if j == 0:
                b.legend(handles=bar_legend(recs), fontsize=7, loc='lower left')
            c = ax[2, j]                                                      # row 3: the same on a linear scale, output power / input power (1 = unchanged)
            for i, r in enumerate(recs):
                off = (i - (n - 1) / 2) * w
                for kk, cc, d in ((key, [mcol.get(m, '#888888') for m in methods], -w / 4), (key + '__control', '#cccccc', w / 4)):
                    mu, se = mean_se(10 ** (z[f'{r}__{kk}'] / 10))
                    c.bar(x + off + d, mu, w / 2 * 0.95, yerr=se, color=cc, hatch=HATCH[i % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
            c.axhline(1, color='k', lw=.6); c.set_ylabel('output power / input power (1 = unchanged)' if j == 0 else '')
            c.set_xticks(x); c.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
        fig.suptitle(f'{ds}: what each removal does to real artifacts and to the brain signal (mean over seeds, ± SE on the bars)', fontweight='bold')
        fig.tight_layout(rect=describe(fig, methods, 0.12)); out = ctx.figure(f'noise_removal_{ds}.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
