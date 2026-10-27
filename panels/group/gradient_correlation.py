"""gradient_correlation: cross-task gradient correlation (EEG-FM-Bench Fig. 3): per backbone parameter group, the task x task cosine
similarity of the tasks' classification gradients on the backbone (tasks = the finetune cells under --head), probe batches paired by index,
mean +- std over --probe-seeds repeats -> gradient_correlation_<group>.png. Near 1: the tasks pull the group the same way (task-agnostic
layer); near 0: unrelated (task-specific). Probe, settings and parameter groups: panels/shared/gradient_probe.py."""
import numpy as np

from panels.shared.groups import own_figure, own_numbers
from panels.core.numbers import load_npz, save_npz
from panels.kit.matrix import matrix_row
from panels.shared import gradient_probe as gp

PRESETS = set()                                  # on request: heavy (backbone gradients of every finetune cell); numbers and figures stay with each backbone


def add_args(flags):
    gp.flags(flags)


def cross_task_cosine(ce):
    """ce [task, seed, batch, group, dim] (unit rows) -> mean, std over seeds of the batch-mean cosine, each [task, task, group]."""
    cos = np.einsum('asbgd,csbgd->acsgb', ce, ce)              # [a, c, seed, group, batch]
    per_seed = np.nanmean(cos, axis=-1)                         # [a, c, seed, group]
    eye = np.eye(ce.shape[0], dtype=bool)[:, :, None, None]
    per_seed = np.where(eye, np.nan, per_seed)
    mean, std = np.nanmean(per_seed, axis=2), np.nanstd(per_seed, axis=2)
    K = ce.shape[2]                                             # diagonal: one task, different batches (the ceiling a pair of tasks can reach)
    own = np.einsum('asbgd,ascgd->asgbc', ce, ce)               # [task, seed, group, batch, batch']
    own = np.where(np.eye(K, dtype=bool), np.nan, own).reshape(*own.shape[:3], -1)
    own = np.nanmean(own, axis=-1)                              # [task, seed, group]
    for a in range(ce.shape[0]):
        mean[a, a], std[a, a] = np.nanmean(own[a], axis=0), np.nanstd(own[a], axis=0)
    return mean, std


def compute(ctx):
    for g in ctx.groups:
        z = gp.gradients(ctx, g)
        mean, std = cross_task_cosine(z['ce'])
        save_npz(own_numbers(ctx, g, 'gradient_correlation.npz'), mean=mean, std=std, tasks=z['tasks'], groups=z['groups'],
                 floor=np.array(1 / np.sqrt(z['ce'].shape[-1])))


def draw(ctx):
    for g in ctx.groups:
        z = load_npz(own_numbers(ctx, g, 'gradient_correlation.npz'))
        tasks, groups = [str(t) for t in z['tasks']], [str(x) for x in z['groups']]
        lim = max(0.2, float(np.nanmax(np.abs(z['mean']))))
        fl = float(z['floor'])
        out = own_figure(ctx, g, 'gradient_correlation.png')
        matrix_row(out, [z['mean'][..., i] for i in range(len(groups))], [z['std'][..., i] for i in range(len(groups))], groups,
                   tasks, tasks, -lim, lim, 'RdBu_r', f'{g}: cross-task gradient correlation (cosine of classification gradients, mean ± std over probe repeats)\n'
                   f'outlined diagonal = the same task on other batches (ceiling); dashed on the colour bar = ±{fl:.3f}, the spread of random directions',
                   'cosine', marks=(-fl, fl), outline_diag=True,
                   note='How to read: the cosine says how much two tasks push a parameter group the same way (1 = the same direction, 0 = unrelated, -1 = opposite).\n'
                        'Single-batch gradients are noisy, so even one task against itself (outlined diagonal) agrees only weakly: judge a cell against the diagonal and the dashed floor, not against 1.')
        print(f'  -> {out}')
