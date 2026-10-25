"""gradient_correlation: cross-task gradient correlation (EEG-FM-Bench Fig. 3): per backbone parameter group, the task x task cosine
similarity of the tasks' classification gradients on the backbone (tasks = the finetune cells under --head), probe batches paired by index,
mean +- std over --probe-seeds repeats -> gradient_correlation_<group>.png. Near 1: the tasks pull the group the same way (task-agnostic
layer); near 0: unrelated (task-specific). Probe, settings and parameter groups: panels/shared/gradient_probe.py."""
import numpy as np

from panels.core.numbers import load_npz, save_npz
from panels.kit.matrix import matrix_row
from panels.shared import gradient_probe as gp

PRESETS = {'standard'}


def add_args(flags):
    gp.flags(flags)


def cross_task_cosine(ce):
    """ce [task, seed, batch, group, dim] (unit rows) -> mean, std over seeds of the batch-mean cosine, each [task, task, group]."""
    cos = np.einsum('asbgd,csbgd->acsgb', ce, ce)              # [a, c, seed, group, batch]
    per_seed = np.nanmean(cos, axis=-1)                         # [a, c, seed, group]
    eye = np.eye(ce.shape[0], dtype=bool)[:, :, None, None]
    per_seed = np.where(eye, np.nan, per_seed)
    return np.nanmean(per_seed, axis=2), np.nanstd(per_seed, axis=2)


def compute(ctx):
    for g in ctx.groups:
        z = gp.gradients(ctx, g)
        mean, std = cross_task_cosine(z['ce'])
        save_npz(ctx.numbers(f'gradient_correlation_{g}.npz'), mean=mean, std=std, tasks=z['tasks'], groups=z['groups'])


def draw(ctx):
    for g in ctx.groups:
        z = load_npz(ctx.numbers(f'gradient_correlation_{g}.npz'))
        tasks, groups = [str(t) for t in z['tasks']], [str(x) for x in z['groups']]
        lim = max(0.2, float(np.nanmax(np.abs(z['mean']))))
        out = ctx.figure(f'gradient_correlation_{g}.png')
        matrix_row(out, [z['mean'][..., i] for i in range(len(groups))], [z['std'][..., i] for i in range(len(groups))], groups,
                   tasks, tasks, -lim, lim, 'RdBu_r', f'{g}: cross-task gradient correlation (cosine of classification gradients, mean ± std over probe repeats)',
                   'cosine')
        print(f'  -> {out}')
