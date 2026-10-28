"""asr_cutoff: the cutoff of the two ASR variants against what they remove and what they damage, per recipe (mean over its seeds): the time-domain ASR (asrpy, cutoffs 3 /
5 / 10 / 20 in standard deviations of the clean window-rms distribution) and ASR in the qlet space (cutoffs 1 / 1.5 / 2 / 3 / 5 times the clean component rms: a ratio, not
the same unit). Row 1: artifact power change in dB against brain kept on clean windows, one point per cutoff, joined in cutoff order. Row 2: the specific part, artifact dB minus
the clean-window control dB (< 0: removes the artifact more than it filters the same band of a clean window), against the cutoff. -> asr_cutoff_<dataset>.png, asr_cutoff.md.
Needs task_filter_artifacts run with --art-asr-cutoffs 3 5 10 --art-atom-cutoffs 1 2 3 5 (numbers artifacts_TUAR / artifacts_TUEV)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz
from panels.pretrain.task_filter_artifacts import COLUMNS
from panels.shared.groups import color_of, db, drop_pooled_muscle, load_all

PRESETS = set()
TIME = (('ASR 3', 3), ('ASR 5', 5), ('ASR 10', 10), ('ASR', 20))
ATOM = (('ASR atoms 1', 1), ('ASR atoms 1.5', 1.5), ('ASR atoms 2', 2), ('ASR atoms 3', 3), ('ASR atoms 5', 5))
MARKERS = ('o', 's', '^', 'D')
TITLE = {'musc__mpair_left': 'muscle: 30-90 Hz power of the annotated pairs', 'musc_elec__mpair_left': 'muscle + electrode: 30-90 Hz power of the annotated pairs', 'eyem__eye_left': 'eye movement: frontal 1-10 Hz power', 'musc__muscle_left': 'muscle: 30-90 Hz power', 'elec__pair_left': 'electrode: annotated pair peak amplitude²',
         'musc_elec__muscle_left': 'muscle + electrode: 30-90 Hz power', 'musc_elec__pair_left': 'muscle + electrode: annotated pair peak amplitude²',
         'musc_elec__peak_left': 'artifact events: peak amplitude²', 'line__line_left': 'real line noise: 59-61 Hz power'}


def load(path):
    z = load_npz(path)
    return {k: z[k] for k in z if '__' in k or k in ('methods', 'brain_kept')}


def curve(v, key, methods, ctl=False):
    """-> per method, the mean over seeds of the dB of its patient-mean (artifact windows, or their clean-window control)."""
    out = []
    for m in methods:
        vals = [db(np.nanmean(s[key + ('__control' if ctl else '')][[str(x) for x in s['methods']].index(m)])) for s in v if m in [str(x) for x in s['methods']]]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def brain(v, methods):
    return np.array([np.mean([np.nanmean(s['brain_kept'][[str(x) for x in s['methods']].index(m)]) for s in v if m in [str(x) for x in s['methods']]] or [np.nan]) for m in methods])


def compute(ctx):
    lines = []
    for ds in ('TUAR', 'TUEV'):
        data = load_all(ctx.groups, load, 'filter', 'multi', f'artifacts_{ds}.npz')
        if not data:
            continue
        have = {m for v in data.values() for s in v for m in map(str, s['methods'])}
        if not any(m.startswith('ASR atoms ') and m != 'ASR atoms 1.5' and 'line' not in m for m in have) and 'ASR 3' not in have:
            print(f'  {ds}: no cutoff sweep in the numbers (task_filter_artifacts --art-asr-cutoffs ... --art-atom-cutoffs ...)'); continue
        cols = [(k, sc) for k, sc in COLUMNS if k != 'eyem_musc' and all(f'{k or "line"}__{sc}'.replace(' ', '_') in s for v in data.values() for s in v)]
        keys = drop_pooled_muscle([f'{k or "line"}__{sc}'.replace(' ', '_') for k, sc in cols])
        out = {'recipes': np.array(list(data)), 'keys': np.array(keys)}
        lines.append(f'\n{ds}: per recipe, per cutoff: artifact dB (control dB) | brain kept')
        for r, v in data.items():
            for name, grid in (('time', TIME), ('atom', ATOM)):
                ms = [m for m, _ in grid]; cs = np.array([c for _, c in grid], float)
                out[f'{r}__{name}__cutoff'] = cs; out[f'{r}__{name}__brain'] = brain(v, ms)
                for key in keys:
                    out[f'{r}__{name}__{key}'] = curve(v, key, ms); out[f'{r}__{name}__{key}__control'] = curve(v, key, ms, True)
                lines.append(f'  {r} {name} ASR')
                for j, (m, c) in enumerate(grid):
                    lines.append(f'    cutoff {c:<4g} ' + ' '.join(f'{out[f"{r}__{name}__{k}"][j]:6.1f} ({out[f"{r}__{name}__{k}__control"][j]:5.1f})' for k in keys) + f' | {out[f"{r}__{name}__brain"][j]:.3f}')
        np.savez(ctx.numbers(f'asr_cutoff_{ds}.npz'), **out)
    if lines:
        open(ctx.numbers('asr_cutoff.md'), 'w').write('```\n' + '\n'.join(lines) + '\n```\n'); print('\n'.join(lines))


def draw(ctx):
    for ds in ('TUAR', 'TUEV'):
        try:
            z = np.load(ctx.numbers(f'asr_cutoff_{ds}.npz'))
        except FileNotFoundError:
            continue
        recs, keys = [str(r) for r in z['recipes']], [str(k) for k in z['keys']]; col = color_of(recs)
        fig, ax = plt.subplots(2, len(keys), figsize=(3.9 * len(keys), 8.2), squeeze=False)
        for j, key in enumerate(keys):
            for i, r in enumerate(recs):
                for name, c, lab in (('time', '#1baf7a', 'time-domain ASR (cutoff in sd)'), ('atom', '#e377c2', 'ASR in the qlet space (cutoff = x clean rms)')):
                    cs = z[f'{r}__{name}__cutoff']; a = z[f'{r}__{name}__{key}']; ctl = z[f'{r}__{name}__{key}__control']; b = z[f'{r}__{name}__brain']; o = np.argsort(cs)
                    ax[0, j].plot(a[o], b[o], color=c, marker=MARKERS[i % len(MARKERS)], ms=6, lw=1.2, label=f'{lab}' + (f', {r}' if len(recs) > 1 else '') if j == 0 else None)
                    for x_, y_, c_ in zip(a[o], b[o], cs[o]):
                        ax[0, j].annotate(f'{c_:g}', (x_, y_), fontsize=6, xytext=(3, 3), textcoords='offset points', color=c)
                    ax[1, j].plot(cs[o], (a - ctl)[o], color=c, marker=MARKERS[i % len(MARKERS)], ms=6, lw=1.2)
            ax[0, j].axvline(0, color='k', lw=.5, ls=':'); ax[0, j].set_xlabel('artifact power change (dB)'); ax[0, j].set_ylabel('brain kept on clean windows' if j == 0 else '')
            ax[0, j].set_title(TITLE.get(key, key), fontsize=9, fontweight='bold')
            ax[1, j].axhline(0, color='k', lw=.6); ax[1, j].set_xscale('log'); ax[1, j].set_xlabel('cutoff (log; time: sd, qlet: x clean rms)')
            ax[1, j].set_ylabel('artifact dB minus control dB (< 0: specific removal)' if j == 0 else '')
        ax[0, 0].legend(fontsize=7, loc='lower left')
        fig.suptitle(f'{ds}: the ASR cutoff of both variants against removal and damage (mean over seeds; the two cutoffs are not the same unit)', fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.96)); out = ctx.figure(f'asr_cutoff_{ds}.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
