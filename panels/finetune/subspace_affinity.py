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


def chance_affinity(batches, rank, dim, draws=200, seed=0):
    """Affinity of two unrelated gradient sets of the same shape (random Gaussian rows): the level that means no relation -> mean, 95th percentile."""
    rng = np.random.default_rng(seed)
    v = [np.linalg.svd(pca_basis(rng.standard_normal((batches, dim)), rank).T @ pca_basis(rng.standard_normal((batches, dim)), rank),
                       compute_uv=False).mean() for _ in range(draws)]
    return float(np.mean(v)), float(np.percentile(v, 95))


def affinity(ce, rank):
    """ce [task, seed, batch, group, dim] -> mean, std of the pairwise affinity, each [task, task, group]: between two tasks over the probe
    repeats; on the diagonal one task against itself (repeat against another repeat: the ceiling)."""
    T, R, K, G, _ = ce.shape
    rank = min(rank, K - 1)
    U = [[[pca_basis(ce[t, r, :, gi], rank) if np.isfinite(ce[t, r, :, gi]).all() else None for gi in range(G)] for r in range(R)] for t in range(T)]
    aff = lambda u, v: np.nan if u is None or v is None else np.linalg.svd(u.T @ v, compute_uv=False).mean()
    mean, std = np.full((T, T, G), np.nan), np.full((T, T, G), np.nan)
    for a in range(T):
        for b in range(T):
            for gi in range(G):
                v = ([aff(U[a][r][gi], U[b][r][gi]) for r in range(R)] if a != b
                     else [aff(U[a][r][gi], U[a][s][gi]) for r in range(R) for s in range(r + 1, R)])
                mean[a, b, gi], std[a, b, gi] = np.nanmean(v), np.nanstd(v)
    return mean, std


def compute(ctx):
    for g in ctx.groups:
        z = gp.gradients(ctx, g)
        mean, std = affinity(z['ce'], ctx.args.affinity_rank)
        rank = min(ctx.args.affinity_rank, z['ce'].shape[2] - 1)
        chance = chance_affinity(z['ce'].shape[2], rank, z['ce'].shape[-1])
        save_npz(ctx.numbers(f'subspace_affinity_{g}.npz'), mean=mean, std=std, tasks=z['tasks'], groups=z['groups'],
                 rank=np.array(rank), chance=np.array(chance))


def draw(ctx):
    for g in ctx.groups:
        z = load_npz(ctx.numbers(f'subspace_affinity_{g}.npz'))
        tasks, groups = [str(t) for t in z['tasks']], [str(x) for x in z['groups']]
        out = ctx.figure(f'subspace_affinity_{g}.png')
        matrix_row(out, [z['mean'][..., i] for i in range(len(groups))], [z['std'][..., i] for i in range(len(groups))], groups,
                   tasks, tasks, 0.0, 1.0, 'YlGnBu',
                   f'{g}: cross-task subspace affinity (rank-{int(z["rank"])} gradient subspaces, mean ± std over probe repeats)\n'
                   f'outlined diagonal = the same task on another probe repeat (ceiling); dashed on the colour bar = {float(z["chance"][0]):.3f}, '
                   f'unrelated subspaces (95th percentile {float(z["chance"][1]):.3f})', 'affinity', marks=(float(z['chance'][0]),), outline_diag=True,
                   note='How to read: each task\'s main gradient directions (a rank-6 basis) are compared across tasks: 0 = unrelated subspaces, 1 = the same subspace.\n'
                        'Cells near the diagonal value share about as much as a task does with itself; cells near the dashed line share nothing.')
        print(f'  -> {out}')
