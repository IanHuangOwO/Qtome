"""subspace_affinity: cross-task subspace affinity (EEG-FM-Bench Fig. 4): per backbone parameter group, how much the tasks' dominant gradient
subspaces overlap -> subspace_affinity_<group>.png. Each task's probe-batch gradients (--probe-batches rows) are centred and reduced to a
--affinity-rank (default 6) PCA basis; the affinity of two tasks is the mean of the top singular values of U^T V (1 = the same subspace, 0 =
orthogonal), mean +- std over --probe-seeds repeats. Probe: panels/shared/gradient_probe.py."""
import numpy as np

from panels.core.numbers import load_npz, save_npz
from panels.kit.matrix import matrix_row
from panels.shared import gradient_probe as gp

PRESETS = {'standard'}


def add_args(flags):
    gp.flags(flags)
    flags.add('--affinity-rank', type=int, default=6, help='PCA rank of every task\'s gradient subspace')


def pca_basis(g, rank):
    """g [batch, dim] -> [dim, rank] orthonormal basis of the top principal directions (rows centred)."""
    g = g - g.mean(0, keepdims=True)
    return np.linalg.svd(g, full_matrices=False)[2][:rank].T


def affinity(ce, rank):
    """ce [task, seed, batch, group, dim] -> mean, std over seeds of the pairwise affinity, each [task, task, group]."""
    T, R, K, G, _ = ce.shape
    rank = min(rank, K - 1)
    out = np.full((T, T, R, G), np.nan)
    for r in range(R):
        for gi in range(G):
            U = [pca_basis(ce[t, r, :, gi], rank) if np.isfinite(ce[t, r, :, gi]).all() else None for t in range(T)]
            for a in range(T):
                for b in range(T):
                    if a != b and U[a] is not None and U[b] is not None:
                        out[a, b, r, gi] = np.linalg.svd(U[a].T @ U[b], compute_uv=False).mean()
    return np.nanmean(out, axis=2), np.nanstd(out, axis=2)


def compute(ctx):
    for g in ctx.groups:
        z = gp.gradients(ctx, g)
        mean, std = affinity(z['ce'], ctx.args.affinity_rank)
        save_npz(ctx.numbers(f'subspace_affinity_{g}.npz'), mean=mean, std=std, tasks=z['tasks'], groups=z['groups'],
                 rank=np.array(min(ctx.args.affinity_rank, z['ce'].shape[2] - 1)))


def draw(ctx):
    for g in ctx.groups:
        z = load_npz(ctx.numbers(f'subspace_affinity_{g}.npz'))
        tasks, groups = [str(t) for t in z['tasks']], [str(x) for x in z['groups']]
        out = ctx.figure(f'subspace_affinity_{g}.png')
        matrix_row(out, [z['mean'][..., i] for i in range(len(groups))], [z['std'][..., i] for i in range(len(groups))], groups,
                   tasks, tasks, 0.0, 1.0, 'YlGnBu',
                   f'{g}: cross-task subspace affinity (rank-{int(z["rank"])} gradient subspaces, mean ± std over probe repeats)', 'affinity')
        print(f'  -> {out}')
