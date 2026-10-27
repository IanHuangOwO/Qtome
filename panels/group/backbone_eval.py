"""backbone_eval: masked reconstruction MSE of every group's backbone on held-out pretrain windows, per mask, mean +- SE over the seeds of each recipe
-> backbone_eval.png and backbone_eval.md (the numbers of the pretrain panel backbone_eval, run on each backbone first). Dots are single seeds; the table
gives each recipe's change against the first recipe."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.shared.groups import color_of, grouped_bars, load_all, mean_se

PRESETS = {'standard'}
MASKS = ('token_runs', 'random_channel', 'channel_cluster', 'motor3_to_bci22', 'time_block', 'time_block_short', 'time_block_3', 'time_block_2')


def compute(ctx):
    data = load_all(ctx.groups, lambda p: json.load(open(p))['ablation_by_mask'], 'backbone_eval.json')
    if not data:
        print('  no backbone_eval numbers: run the pretrain panel backbone_eval on each backbone'); return
    masks = [m for m in MASKS if all(m in s for v in data.values() for s in v)]
    rows = {r: np.array([[s[m]['baseline'] for m in masks] for s in v]) for r, v in data.items()}
    np.savez(ctx.numbers('backbone_eval.npz'), masks=np.array(masks), recipes=np.array(list(rows)), **{f'mse__{r}': v for r, v in rows.items()})
    ref = next(iter(rows)); lines = ['| mask | ' + ' | '.join(f'{r} (n={len(v)})' for r, v in rows.items()) + ' |', '|---|' + '---|' * len(rows)]
    for j, m in enumerate(masks):
        cells = []
        for r, v in rows.items():
            mu, se = mean_se(v[:, j:j + 1]); ch = '' if r == ref else f' ({100 * (mu[0] / rows[ref][:, j].mean() - 1):+.1f}%)'
            cells.append(f'{mu[0]:.4f} ± {se[0]:.4f}{ch}')
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    open(ctx.numbers('backbone_eval.md'), 'w').write('\n'.join(lines) + f'\n\nMasked MSE on held-out pretrain windows, mean ± SE over seeds; change against {ref}.\n')
    print('\n'.join(lines))


def draw(ctx):
    z = np.load(ctx.numbers('backbone_eval.npz'))
    masks, recs = [str(m) for m in z['masks']], [str(r) for r in z['recipes']]; col = color_of(recs)
    fig, ax = plt.subplots(figsize=(1.3 * len(masks) + 2, 4.2))
    grouped_bars(ax, masks, {r: mean_se(z[f'mse__{r}']) for r in recs}, col)
    for i, r in enumerate(recs):
        v = z[f'mse__{r}']; w = 0.8 / len(recs)
        for s in v:
            ax.scatter(np.arange(len(masks)) + (i - (len(recs) - 1) / 2) * w, s, color='k', s=6, zorder=3)
    ax.set_ylabel('masked MSE (lower is better)'); ax.legend(fontsize=8)
    ax.set_title('Masked reconstruction on held-out pretrain windows: mean ± SE over seeds (dots = seeds)', fontsize=10, fontweight='bold')
    fig.tight_layout(); out = ctx.figure('backbone_eval.png'); fig.savefig(out, dpi=130); plt.close(fig); print(f'  -> {out}')
