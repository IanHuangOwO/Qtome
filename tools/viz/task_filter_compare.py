"""The artifact comparison plot (the task_filter_compare panel; numbers from tools/analysis/task_filter_eval.compare). One
column per artifact. Top: the paradigm's lda on the cleaned trials against the gain (signal kept over artifact kept,
dB). Bottom: signal kept against artifact kept. Each task filter is a line over its shares (labelled at the points);
input, the 64-channel reconstruction without filter, ASR and ICLabel are single points. Mean over subjects; the dashed
line is the clean recording's lda."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINES = {'single-task': '#2a78d6', 'multi-task': '#eb6834'}             # categorical slots 1, 2
POINTS = {'input': ('#555555', 's'), 'backbone64': ('#555555', '^'), 'ASR': ('#1baf7a', '*'), 'ICLabel': ('#eda100', 'D')}


def draw(rows, dst, sources, out_path):
    """rows from compare(); sources {label: what it was built from}, for the legend."""
    metric = rows['clean']['metric']
    kinds = [k for k in rows if k != 'clean']; m_ = lambda r, k: float(np.mean(r[k]))
    fig, ax = plt.subplots(2, len(kinds), figsize=(5.2 * len(kinds), 9.5), squeeze=False)
    for j, kind in enumerate(kinds):
        r = rows[kind]
        for lab, col in LINES.items():
            ks = sorted((k for k in r if k.startswith(lab + ' ')), key=lambda k: -float(k.split()[-1]))   # shares, large to small
            if not ks: continue
            for a_, xs, ys in ((ax[0, j], [m_(r[k], 'gain dB') for k in ks], [m_(r[k], 'lda') for k in ks]),
                               (ax[1, j], [m_(r[k], 'artifact') for k in ks], [m_(r[k], 'signal') for k in ks])):
                a_.plot(xs, ys, 'o-', color=col, lw=2, ms=8, mec='white', mew=1.5, label=f'{lab} ({sources[lab]})')
                for k, x_, y_ in zip(ks, xs, ys): a_.annotate(k.split()[-1], (x_, y_), fontsize=9, color='#333333', xytext=(5, 4), textcoords='offset points')
        for m, (col, mk) in POINTS.items():
            if m not in r: continue
            lab = m + (f' ({m_(r[m], "removed"):.1f} components)' if m == 'ICLabel' and 'removed' in r[m] else '')
            ax[0, j].plot(m_(r[m], 'gain dB'), m_(r[m], 'lda'), mk, color=col, ms=11, mec='white', mew=1, label=lab)
            ax[1, j].plot(m_(r[m], 'artifact'), m_(r[m], 'signal'), mk, color=col, ms=11, mec='white', mew=1, label=lab)
        ax[0, j].axhline(m_(rows['clean']['raw'], 'lda'), color='#999999', ls='--', lw=1)
        ax[0, j].set_title(kind, fontsize=13); ax[0, j].set_xlabel('gain, dB (signal kept / artifact kept)'); ax[0, j].set_ylabel(metric)
        ax[1, j].plot([0, 1.4], [0, 1.4], color='#cccccc', lw=.8, zorder=0); ax[1, j].set_xlabel('artifact kept'); ax[1, j].set_ylabel('signal kept')
        for a_ in ax[:, j]:
            a_.grid(color='#eeeeee', lw=.8); a_.spines[['top', 'right']].set_visible(False)
    ax[0, 0].legend(fontsize=8, frameon=False); ax[1, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{dst} (never seen by either filter): artifacts at 50% of trial power; numbers = share of cells kept; '
                 f'dashed = clean recording', fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)
