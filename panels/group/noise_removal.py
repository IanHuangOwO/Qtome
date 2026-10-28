"""noise_removal: what each removal method does to the artifacts of TUAR and TUEV, and to the clean brain signal, per recipe: mean +- SE over the seeds of
each recipe -> noise_removal_<dataset>.png and noise_removal.md. Bars: artifact power change in dB (0 = unchanged, lower = more removed) next to grey bars: the control (the
same method on clean windows: a method that removes the artifact and the control equally is a frequency filter); top row: brain kept (correlation with the
clean windows) against the artifact power change, one marker shape per recipe; bottom row: the same on a linear scale (output power / input power). Needs the pretrain panel task_filter_artifacts (TUAR and TUEV) and, for atom removal, atom_labels on each backbone."""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz
from panels.pretrain.task_filter_artifacts import ATOM_COLORS, COLUMNS, TUEV_KINDS
from panels.shared.groups import bar_legend, color_of, column_methods, describe, db, drop_pooled_muscle, grouped_bars, load_all, mean_se

PRESETS = {'standard'}
HOW = {'TUAR': ("How the artifacts are defined. Artifact windows: 5 s windows of one patient that the annotation covers for at least half with the labelled event (eyem eye movement, musc muscle, elec electrode; a mixed label such as "
                "musc_elec is a window with that mixed event) and for less than 10% with any other label. Clean windows (grey bars, the control): the same patient's windows with no annotated event at all. Real recordings have no "
                "separate noise signal, so each column scores one measure of the whole window (brain plus artifact), on the method's output and on its input (windows scaled by their own standard deviation): eye = 1-10 Hz power "
                "of the mean of Fp1 and Fp2; muscle = 30-90 Hz power of the annotated bipolar pairs (a - b), the channels the annotator marked; electrode = peak amplitude squared (99.9th percentile of |a - b|) of the annotated bipolar pair; line noise = 59-61 Hz power on clean windows "
                "(its control is 8-30 Hz power). Row 3 is that measure's power, output / input: 1 = unchanged, 0 = that band is gone from the window (brain and artifact together, not the noise alone). Top row: brain kept = correlation "
                "of the output with the input on clean windows."),
       'TUEV': ("How the artifacts are defined. Artifact windows: the windows of the TUEV event labelled eyem (eye movement) or artf (a generic artifact, scored on muscle and peak amplitude); clean windows (grey bars, the control): "
                "the windows labelled background. Real recordings have no separate noise signal, so each column scores one measure of the whole window (brain plus artifact), on the method's output and on its input (windows scaled by "
                "their own standard deviation): eye = 1-10 Hz power of the mean of Fp1 and Fp2; muscle = 30-90 Hz power over all channels; peak = peak amplitude squared (99.9th percentile of |amplitude| over all channels), pooled, as TUEV "
                "has no per-pair annotation here; line noise = 59-61 Hz power on clean windows (its control is 8-30 Hz power). Row 3 is that measure's power, output / input: 1 = unchanged, 0 = that band is gone from the window (brain and "
                "artifact together, not the noise alone). Top row: brain kept = correlation of the output with the input on clean windows.")}
MARKERS = ('o', 's', '^', 'D')
HATCH = ('', '//', '..', 'xx')
ATOM_ASR = ('ASR atoms 1.5', 'ASR atoms 1.5 + atoms -line')                # from the pretrain panel task_filter_atom_asr (--asr-tag _combo), same patients as task_filter_artifacts
SHOW = ('backbone64', 'multi-task 0.5', 'multi-task 0.1', 'atoms -eye', 'atoms -muscle', 'atoms -line', 'atoms -artifacts', 'atoms -brain', 'atoms brain only', 'ASR', 'ICLabel', *ATOM_ASR)
WHAT = {'eye left': 'frontal 1-10 Hz power', 'muscle left': '30-90 Hz power', 'peak left': 'peak amplitude²', 'pair left': 'peak amplitude² of the annotated bipolar pair', 'mpair left': '30-90 Hz power of the annotated pairs', 'line left': '59-61 Hz power (real line noise)'}
NAME = {'eyem': 'eye movement', 'musc': 'muscle', 'elec': 'electrode', 'eyem_musc': 'eyem + musc', 'musc_elec': 'musc + elec', None: 'line noise'}


def load(path):
    z = load_npz(path)
    out = {k: z[k] for k in z if '__' in k or k in ('methods', 'brain_kept')}
    extra = path.replace('artifacts_TUAR.npz', 'atom_asr_TUAR_combo.npz')
    if extra != path and os.path.exists(extra):
        e = load_npz(extra); em = [str(m) for m in e['methods']]; have = [str(m) for m in out['methods']]; rows = [em.index(m) for m in ATOM_ASR if m in em and m not in have]
        if rows:
            for k in out:
                if k != 'methods' and k in e:
                    n = max(out[k].shape[1], e[k].shape[1]); pad = lambda a: np.pad(a, ((0, 0), (0, n - a.shape[1])), constant_values=np.nan)
                    out[k] = np.concatenate([pad(out[k]), pad(e[k][rows])])
            out['methods'] = np.array([*(str(m) for m in out['methods']), *(em[i] for i in rows)])
    return out


def add_args(flags):
    flags.add('--with-multitask', action='store_true', default=False, help='noise_removal / noise_removal_synthetic / noise_classify: draw the earlier version too, every method in every column, the multi-task filter included (<name>_with_multitask.png)')


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
    for full in (False, True) if ctx.args.with_multitask else (False,):
        draw_all(ctx, full)


def draw_all(ctx, full):
    for ds in ('TUAR', 'TUEV'):
        try:
            z = np.load(ctx.numbers(f'noise_removal_{ds}.npz'))
        except FileNotFoundError:
            continue
        methods, cols, recs = [str(m) for m in z['methods']], drop_pooled_muscle([str(c) for c in z['cols']]), [str(r) for r in z['recipes']]; col = color_of(recs); n = len(recs)
        mcol = {'backbone64': '#555555', 'ASR': '#1baf7a', 'ICLabel': '#eda100', 'multi-task 0.5': '#eb6834', 'multi-task 0.1': '#b3410f', **{m: c for m, (c, _) in ATOM_COLORS.items()}, 'ASR atoms 1.5': '#e377c2', 'ASR atoms 1.5 + atoms -line': '#d62728'}
        fig, ax = plt.subplots(3, len(cols), figsize=(3.6 * len(cols), 13.5), squeeze=False); w = 0.8 / n
        shown = methods if full else [m for m in methods if any(m in column_methods(methods, c.split('__')[0]) for c in cols)]
        for j, key in enumerate(cols):
            k, sc = key.split('__', 1); a = ax[0, j]; ms = methods if full else column_methods(methods, k); ix = [methods.index(m) for m in ms]; x = np.arange(len(ms))
            for i, r in enumerate(recs):                                      # row 1: removal against damage, one marker shape per recipe, one colour per method
                mk = MARKERS[i % len(MARKERS)]
                for m in ms:
                    a.scatter(z[f'{r}__{key}'][:, methods.index(m)].mean(), z[f'{r}__brain'][:, methods.index(m)].mean(), color=mcol.get(m, '#888888'), marker=mk, s=60, zorder=3)
            a.scatter([0], [1], color='#555555', marker='s', s=60, zorder=3); a.axvline(0, color='k', lw=.5, ls=':')
            a.set_xlabel('artifact power change (dB)'); a.set_ylabel('brain kept on clean windows (correlation)' if j == 0 else '')
            a.set_title(f'{NAME[None if k == "line" else k]}\n{WHAT[sc.replace("_", " ")]}', fontsize=9, fontweight='bold')
            if j == 0:
                h = [plt.Line2D([], [], marker='s', color='#555555', ls='', label='input')] + [plt.Line2D([], [], marker='o', color=c, ls='', label=m) for m, c in mcol.items() if m in shown]
                h += [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], color='k', mfc='none', ls='', label=f'recipe {r}') for i, r in enumerate(recs)]
                a.legend(handles=h, fontsize=7, loc='lower left')
            b = ax[1, j]                                                      # row 2: artifact windows (method colour; hatch = recipe) next to the control on clean windows (grey)
            for i, r in enumerate(recs):
                off = (i - (n - 1) / 2) * w
                for kk, c, d in ((key, [mcol.get(m, '#888888') for m in ms], -w / 4), (key + '__control', '#cccccc', w / 4)):
                    mu, se = mean_se(z[f'{r}__{kk}'][:, ix])
                    b.bar(x + off + d, mu, w / 2 * 0.95, yerr=se, color=c, hatch=HATCH[i % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
            b.axhline(0, color='k', lw=.6); b.set_ylabel('power of the scored measure, change (dB)\n(0 = unchanged; grey = clean windows)' if j == 0 else '')
            b.set_xticks(x); b.set_xticklabels(ms, rotation=45, ha='right', fontsize=7)
            if j == 0:
                b.legend(handles=bar_legend(recs), fontsize=7, loc='lower left')
            c = ax[2, j]                                                      # row 3: the same on a linear scale, output power / input power (1 = unchanged)
            for i, r in enumerate(recs):
                off = (i - (n - 1) / 2) * w
                for kk, cc, d in ((key, [mcol.get(m, '#888888') for m in ms], -w / 4), (key + '__control', '#cccccc', w / 4)):
                    mu, se = mean_se(10 ** (z[f'{r}__{kk}'][:, ix] / 10))
                    c.bar(x + off + d, mu, w / 2 * 0.95, yerr=se, color=cc, hatch=HATCH[i % len(HATCH)], edgecolor='white', linewidth=0, capsize=1.5, error_kw={'lw': 0.8})
            c.axhline(1, color='k', lw=.6); c.set_ylabel('power of the scored measure, output / input\n(1 = unchanged, 0 = band gone)' if j == 0 else '')
            c.set_xticks(x); c.set_xticklabels(ms, rotation=45, ha='right', fontsize=7)
        fig.suptitle(f'{ds}: what each removal does to real artifacts and to the brain signal (mean over seeds, ± SE on the bars). Rows 2 and 3: power of the scored measure, output / input' + (': every method, multi-task filter included' if full else ''), fontweight='bold')
        fig.text(0.01, 0.962, textwrap.fill(HOW[ds], int(40 * len(cols))), fontsize=8, va='top', ha='left', color='#333333'); fig.tight_layout(rect=(lambda r: (r[0], r[1], r[2], 0.89))(describe(fig, shown, 0.12))); out = ctx.figure(f'noise_removal_{ds}{"_with_multitask" if full else ""}.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
