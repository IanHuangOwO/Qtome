"""pretrain_alignment: gradient alignment between pretraining and downstream tasks (EEG-FM-Bench Fig. 8): per task and backbone parameter
group, the cosine between the reconstruction-loss gradient and the classification-loss gradient on the same trials (left) and the share of
probe batches where it is negative, a conflict (right) -> pretrain_alignment_<group>.png. Mean +- std over --probe-seeds repeats.
Cosine near 0 or negative: reconstruction does not push the group where classification does. Probe: panels/shared/gradient_probe.py."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panels.shared.groups import own_figure, own_numbers
from panels.core.numbers import load_npz, save_npz
from panels.kit.matrix import annotated_matrix, colorbar_marks
from panels.shared import gradient_probe as gp

PRESETS = set()                                  # on request: heavy (backbone gradients of every finetune cell); numbers and figures stay with each backbone


def add_args(flags):
    gp.flags(flags)


def alignment(ce, rec):
    """[task, seed, batch, group, dim] x2 -> (cosine mean, std, conflict share mean, std), each [task, group]; stats over seeds."""
    cos = (ce * rec).sum(-1)                                    # [task, seed, batch, group]
    c, conflict = np.nanmean(cos, axis=2), np.nanmean(np.where(np.isnan(cos), np.nan, cos < 0), axis=2)
    return np.nanmean(c, 1), np.nanstd(c, 1), np.nanmean(conflict, 1), np.nanstd(conflict, 1)


def compute(ctx):
    for g in ctx.groups:
        z = gp.gradients(ctx, g)
        cm, cs, pm, ps = alignment(z['ce'], z['rec'])
        save_npz(own_numbers(ctx, g, 'pretrain_alignment.npz'), cos_mean=cm, cos_std=cs, conflict_mean=pm, conflict_std=ps,
                 tasks=z['tasks'], groups=z['groups'], floor=np.array(1 / np.sqrt(z['ce'].shape[-1])))


def draw(ctx):
    for g in ctx.groups:
        z = load_npz(own_numbers(ctx, g, 'pretrain_alignment.npz'))
        tasks, groups = [str(t) for t in z['tasks']], [str(x) for x in z['groups']]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(2.0 * len(groups) + 3, 0.5 * len(tasks) + 3.4))
        lim = max(0.2, float(np.nanmax(np.abs(z['cos_mean'])))); fl = float(z['floor'])
        im1 = annotated_matrix(a1, z['cos_mean'], z['cos_std'], tasks, groups, -lim, lim, 'RdBu_r', 'cosine(reconstruction, classification)')
        im2 = annotated_matrix(a2, z['conflict_mean'], z['conflict_std'], tasks, groups, 0.0, 1.0, 'YlGnBu',
                               'share of batches in conflict (cosine < 0)')
        colorbar_marks(fig.colorbar(im1, ax=a1, fraction=0.04), (-fl, fl)); colorbar_marks(fig.colorbar(im2, ax=a2, fraction=0.04), (0.5,))
        fig.suptitle(f'{g}: reconstruction vs classification gradients on the same trials\n'
                     f'dashed on the colour bars: ±{fl:.3f} = the spread of random directions (left), 0.5 = chance of a negative cosine (right)', fontweight='bold')
        fig.text(0.5, -0.02, 'How to read: left, the cosine between the reconstruction-loss gradient and the classification-loss gradient on the same trials (> 0: reconstruction also helps classification; 0: unrelated; < 0: opposed).\n'
                 'A pretrained backbone sits near a reconstruction minimum, so that gradient is mostly noise: values inside the dashed lines cannot be told from random. Right, how often the cosine is negative (0.5 = chance).',
                 ha='center', va='top', fontsize=8, color='#333333')
        out = own_figure(ctx, g, 'pretrain_alignment.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')
