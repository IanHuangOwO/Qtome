"""task_filter_compare: the multi-task filter (and, with --single <dataset>, that dataset's single-task filter) on each --target with injected
artifacts, against ASR and ICLabel -> analysis/task_filter/compare/compare_<target>.json and .log, analysis/pretrain/task_filter/compare/
compare_<target>.png. Builds either filter first if it is not saved.

The figure: One
column per artifact. Top: the paradigm's lda on the cleaned trials against the gain (signal kept over artifact kept,
dB). Second row: signal kept against artifact kept. Third to fifth rows: gain, SDR and lda as bars, one per
method and share, mean +- standard error over subjects. Each task filter is a line over its shares (labelled at the points);
input, the 64-channel reconstruction without filter, ASR and ICLabel are single points. Mean over subjects; the dashed
line is the clean recording's lda."""
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.shared.task_filter.context import filter_context, write_table
from panels.shared.task_filter.tests import POWER, compare

PRESETS = set()


def add_args(flags):
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--target', nargs='+', help='unseen datasets (required)')
    flags.add('--single', help='optional: source dataset of a single-task filter to add (not a target); default: the multi-task filter only')


LINES = {'single-task': '#2a78d6', 'multi-task': '#eb6834'}             # categorical slots 1, 2


POINTS = {'input': ('#555555', 's'), 'backbone64': ('#555555', '^'), 'ASR': ('#1baf7a', '*'), 'ICLabel': ('#eda100', 'D')}


def sdr(r):
    """Per subject, dB: -10 log10 of the error of the cleaned trial against the clean one, relative to the clean signal,
    (1 - s)^2 + POWER a^2 for signal kept s and artifact kept a (the artifact has POWER of the signal's power; the part of the
    cleaned trial that neither explains is left out, so it is an upper bound). The input is 3 dB; floored at 30 dB."""
    s, a = np.asarray(r['signal'], float), np.asarray(r['artifact'], float)
    return -10 * np.log10(np.maximum((1 - s) ** 2 + POWER * a ** 2, 1e-3))


def draw_figure(rows, dst, sources, out_path, seen=False):
    """rows from compare(); sources {label: what it was built from}, for the legend."""
    sc = 'loso' if 'loso' in rows['clean']['raw'] else 'lda'              # leave-one-subject-out score when the numbers have it
    metric = rows['clean']['metric'].replace('shrinkage LDA', 'LOSO shrinkage LDA' if sc == 'loso' else 'shrinkage LDA')
    kinds = [k for k in rows if k != 'clean']; m_ = lambda r, k: float(np.mean(r[k]))
    fig, ax = plt.subplots(5, len(kinds), figsize=(5.2 * len(kinds), 23.5), squeeze=False)
    for j, kind in enumerate(kinds):
        r = rows[kind]
        for lab, col in LINES.items():
            ks = sorted((k for k in r if k.startswith(lab + ' ')), key=lambda k: -float(k.split()[-1]))   # shares, large to small
            if not ks: continue
            for a_, xs, ys in ((ax[0, j], [m_(r[k], 'gain dB') for k in ks], [m_(r[k], sc) for k in ks]),
                               (ax[1, j], [m_(r[k], 'artifact') for k in ks], [m_(r[k], 'signal') for k in ks])):
                a_.plot(xs, ys, 'o-', color=col, lw=2, ms=8, mec='white', mew=1.5, label=f'{lab} ({sources[lab]})')
                for k, x_, y_ in zip(ks, xs, ys): a_.annotate(k.split()[-1], (x_, y_), fontsize=9, color='#333333', xytext=(5, 4), textcoords='offset points')
        for m, (col, mk) in POINTS.items():
            if m not in r: continue
            lab = m + (f' ({m_(r[m], "removed"):.1f} components)' if m == 'ICLabel' and 'removed' in r[m] else '')
            ax[0, j].plot(m_(r[m], 'gain dB'), m_(r[m], sc), mk, color=col, ms=11, mec='white', mew=1, label=lab)
            ax[1, j].plot(m_(r[m], 'artifact'), m_(r[m], 'signal'), mk, color=col, ms=11, mec='white', mew=1, label=lab)
        ax[0, j].axhline(m_(rows['clean']['raw'], sc), color='#999999', ls='--', lw=1)
        ax[0, j].set_title(kind, fontsize=13); ax[0, j].set_xlabel('gain, dB (signal kept / artifact kept)'); ax[0, j].set_ylabel(metric)
        ax[1, j].plot([0, 1.4], [0, 1.4], color='#cccccc', lw=.8, zorder=0); ax[1, j].set_xlabel('artifact kept'); ax[1, j].set_ylabel('signal kept')
        for a_ in ax[:, j]:
            a_.grid(color='#eeeeee', lw=.8); a_.spines[['top', 'right']].set_visible(False)
    ax[0, 0].legend(fontsize=8, frameon=False); ax[1, 0].legend(fontsize=8, frameon=False)
    order = ['input', 'backbone64', 'ICLabel', 'ASR'] + [f'{lab} {q}' for lab in LINES for q in ('0.5', '0.25', '0.1')]
    for j, kind in enumerate(kinds):                                      # bars: gain and lda per method, error bar = SE over subjects
        r = rows[kind]; ks = [k for k in order if k in r]
        col = [LINES.get(k.rsplit(' ', 1)[0], POINTS.get(k, ('#555555',))[0]) for k in ks]
        for a_, key in ((ax[2, j], 'gain dB'), (ax[3, j], 'SDR'), (ax[4, j], sc)):
            v = [sdr(r[k]) if key == 'SDR' else np.asarray(r[k][key], float) for k in ks]
            a_.bar(range(len(ks)), [x.mean() for x in v], yerr=[x.std(ddof=1) / len(x) ** .5 for x in v], color=col, capsize=3, edgecolor='white')
            a_.set_xticks(range(len(ks))); a_.set_xticklabels(ks, rotation=60, ha='right', fontsize=8); a_.grid(axis='y', color='#eeeeee', lw=.8)
            a_.spines[['top', 'right']].set_visible(False); a_.set_ylabel(metric if key == sc else key + ', dB')
        ax[2, j].axhline(0, color='k', lw=.8); ax[3, j].axhline(sdr(r['input']).mean(), color='#999999', ls='--', lw=1); ax[4, j].axhline(m_(rows['clean']['raw'], sc), color='#999999', ls='--', lw=1)
        lo = min(m_(r[k], sc) for k in ks); ax[4, j].set_ylim(max(0, lo - .05), None)
    who = 'one of the multi-task filter sources' if seen else 'never seen by either filter'
    fig.suptitle(f'{dst} ({who}): artifacts at 50% of trial power; numbers = share of cells kept; '
                 f'dashed = clean recording', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(out_path, dpi=100); plt.close(fig)


def compute(ctx):
    if not ctx.args.target:
        raise ValueError('task_filter_compare needs --target <dataset> [...]')
    ctxs = {'single-task': filter_context(ctx, ctx.args.single)[1]} if ctx.args.single else {}   # the single-task filter is optional: it lost to the multi-task one
    ctxs['multi-task'] = filter_context(ctx, 'multi')[1]
    sources = ctx.args.filter_sources or list(ctxs['multi-task'].filt['sources'])
    for dst in ctx.args.target:
        assert dst != ctx.args.single, f'{dst} built the single-task filter: not an unseen target'
        seen = dst in sources                                             # a source of the multi-task filter: in-sample for it, and the figure says so
        lines = []
        log = lambda s: (print(s, flush=True), lines.append(s))
        rows = compare(ctxs, dst, log=log)
        stem = ctx.numbers('task_filter', 'compare', f'compare_{dst}')
        write_table(stem, rows, lines)
        with open(stem + '.meta.json', 'w') as f:        # what the figure's legend and title need besides the rows
            json.dump({'sources': {**({'single-task': ctx.args.single} if ctx.args.single else {}), 'multi-task': ' + '.join(sources)},
                       'seen': seen}, f)


def draw(ctx):
    for dst in ctx.args.target:
        stem = ctx.numbers('task_filter', 'compare', f'compare_{dst}')
        rows, meta = json.load(open(stem + '.json')), json.load(open(stem + '.meta.json'))
        draw_figure(rows, dst, meta['sources'], ctx.figure('task_filter', 'compare', f'compare_{dst}.png'), seen=meta['seen'])
