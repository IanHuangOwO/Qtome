"""Groups of backbones for the group stage: a group name is '<recipe>_s<seed>' (free_s1, fixed_s2) or a bare 's1'; the recipe is the name without its seed,
so several recipes can be compared in one run (--group free_s1=bb --group free_s2=bb --group fixed_s1=bb ...). The new group panels read the numbers the
pretrain panels saved under each backbone (output/Qlet/<backbone>/pretrain/analysis/...) and give mean +- SE over the seeds of each recipe."""
import os
import re

import textwrap

import numpy as np

from IO.config import QLET_OUTPUT

PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#9a5fd0', '#eda100', '#888888')


def recipe_of(name):
    m = re.match(r'^(.*?)[_-]?s\d+$', name)
    return name if not m else (m.group(1) or 'seeds')


def recipes(groups):
    """{group: backbone} -> {recipe: [(group, backbone), ...]} in the order given."""
    out = {}
    for g, bb in groups.items():
        out.setdefault(recipe_of(g), []).append((g, bb))
    return out


def numbers_path(bb, *parts):
    return os.path.join(QLET_OUTPUT, bb, 'pretrain', 'analysis', *parts)


def load_all(groups, loader, *parts):
    """{recipe: [loader(path) of every seed whose numbers exist]}; a seed without them is reported and left out."""
    out = {}
    for r, members in recipes(groups).items():
        out[r] = []
        for g, bb in members:
            p = numbers_path(bb, *parts)
            if os.path.exists(p):
                out[r].append(loader(p))
            else:
                print(f'  {g} ({bb}): no {os.path.join(*parts)}; left out')
    return {r: v for r, v in out.items() if v}


def mean_se(rows):
    """rows [seeds, ...] -> (mean, SE over seeds); SE is 0 with one seed."""
    a = np.asarray(rows, dtype=float)
    return np.nanmean(a, 0), (np.nanstd(a, 0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(a.shape[1:]))


def db(r, floor=-40.0):
    return np.maximum(10 * np.log10(np.maximum(r, 1e-12)), floor)


def color_of(recipe_names):
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(recipe_names)}


def grouped_bars(ax, labels, series, colors, width=0.8, **kw):
    """series {recipe: (mean [n], se [n])} -> bars side by side per label."""
    n = len(series); w = width / max(n, 1); x = np.arange(len(labels))
    for i, (r, (mu, se)) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, mu, w * 0.95, yerr=se, color=colors[r], capsize=1.5, error_kw={'lw': 0.8}, label=r, **kw)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)


def own_numbers(ctx, group, name):
    """A single backbone's finetune-side numbers, kept with the backbone: output/Qlet/<backbone>/finetune/_analysis/<head>/<name>
    (not inside finetune/<head>/, whose subfolders are the finetune cells)."""
    path = os.path.join(QLET_OUTPUT, ctx.groups[group], 'finetune', '_analysis', ctx.head, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def own_figure(ctx, group, name):
    """A single backbone's finetune-side figure: output/Qlet/<backbone>/analysis/finetune/<head>/<name> (next to its training dashboards)."""
    path = os.path.join(QLET_OUTPUT, ctx.groups[group], 'analysis', 'finetune', ctx.head, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


METHOD_TEXT = {
    'backbone64': 'reconstruction through the encoder and all qlets, nothing removed (the cost of the model itself)',
    'multi-task 0.5': 'task filter keeping the 50% of the signal that the motor-imagery, P300 and SSVEP heads use',
    'multi-task 0.1': 'task filter keeping only the 10% of the signal the heads use most',
    'atoms -eye': 'reconstruction without the qlets whose top ICLabel class is eye',
    'atoms -muscle': 'without the qlets labelled muscle',
    'atoms -line': 'without the qlets labelled line noise or channel noise',
    'atoms -artifacts': 'without every qlet labelled eye, muscle, heart, line noise or channel noise',
    'atoms -brain': 'without the qlets labelled brain (negative control: it should damage the brain signal)',
    'atoms brain only': 'keeps only the qlets labelled brain',
    'ASR': 'artifact subspace reconstruction, calibrated on the same patient\'s clean windows',
    'ICLabel': 'ICA, then components labelled as artifact by ICLabel are removed',
    'ASR atoms 1.5': 'ASR on the qlet gains: per atom, channel components stronger than 1.5 x the patient\'s clean rms are set to zero',
    'ASR atoms 1.5 + atoms -line': 'that ASR, then without the qlets labelled line noise or channel noise',
}


GENERAL = ('backbone64', 'ASR', 'ICLabel', 'ASR atoms 1.5', 'ASR atoms 1.5 + atoms -line')   # shown in every noise column
MATCHED = {'eyem': 'atoms -eye', 'eye': 'atoms -eye', 'musc': 'atoms -muscle', 'muscle': 'atoms -muscle', 'elec': 'atoms -artifacts', 'electrode': 'atoms -artifacts', 'line': 'atoms -line'}


def column_methods(methods, kind):
    """The methods drawn in the column of one noise kind ('eyem_musc' = both): the general ones and the atom removal that matches the noise. The multi-task
    filter and the other atom groups are left out of the figures (their numbers stay in the .md tables)."""
    want = {MATCHED[t] for t in kind.split('_') if t in MATCHED}
    return [m for m in methods if m in GENERAL or m in want]


def describe(fig, methods, top=0.14):
    """Method legend at the bottom of a figure (call before tight_layout, which then keeps `top` of the height free)."""
    lines = [f'{m}: {METHOD_TEXT[m]}' for m in methods if m in METHOD_TEXT]
    half = (len(lines) + 1) // 2
    for k, chunk in enumerate((lines[:half], lines[half:])):
        fig.text(0.01 + 0.5 * k, 0.005, '\n'.join(textwrap.fill(t, 95, subsequent_indent='    ') for t in chunk), fontsize=7.5, va='bottom', ha='left')
    return (0, top, 1, 0.97)


def bar_legend(recipes, hatch=('', '//', '..', 'xx')):
    """Legend handles for bars coloured by method: one grey hatched patch per recipe (only with several) and the grey control bar."""
    from matplotlib.patches import Patch
    h = [Patch(facecolor='#888888', hatch=hatch[i % len(hatch)], edgecolor='white', label=r) for i, r in enumerate(recipes)] if len(recipes) > 1 else []
    return h + [Patch(facecolor='#cccccc', label='control (clean windows)')]
