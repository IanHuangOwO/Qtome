"""atom_distribution: per-atom a / b / amplitude / phase violins over real trials, one plot per dataset
-> analysis/pretrain/atom_distribution/<dataset>.png. Options: --datasets (default: every pretrain dataset), --max-trials (per dataset)."""
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from IO.dataset import build_dataset_from_config
from panels.core.numbers import load_npz, save_npz
from panels.shared.trials import cap_subjects

PRESETS = {'standard'}


def add_args(flags):
    flags.add('--datasets', nargs='+', help='these pretrain datasets (default: every one)')
    flags.add('--max-trials', type=int, default=1000, help='trials per dataset')


@torch.no_grad()
def accumulate_atom_ab(model, pretrain_dataset, trial_indices, device, max_atoms=30):
    """Runs the frozen tokenizer (no masking) over pretrain_dataset[trial_indices] and collects every
    (patch, channel)'s post-rms (a, b) pair per Q-atom (AtomBank's quadrature gain). Returns
    {atom_id: (a [n], b [n])} np.ndarray pairs for the max_atoms Q-atoms with the largest mean
    amplitude, in that order."""
    was_training = model.training
    model.eval()
    amps = []
    try:
        for trial_idx in trial_indices:
            x_patches, coords, _mask, time_indices, _y, valid_channels = pretrain_dataset[trial_idx]
            out = model.encode_atoms(x_patches.unsqueeze(0).to(device), coords.unsqueeze(0).to(device),
                                      time_idx=time_indices.unsqueeze(0).to(device),
                                      valid_channels=valid_channels.unsqueeze(0).to(device))
            amps.append(out.amp[:, valid_channels.to(device).bool()].reshape(-1, model.n_atoms, 2).cpu())
    finally:
        model.train(was_training)
    ab = torch.cat(amps)                                                   # [n, S, 2]
    order = ab.norm(dim=-1).mean(0).argsort(descending=True)[:max_atoms].tolist()
    return {s: (ab[:, s, 0].numpy(), ab[:, s, 1].numpy()) for s in order}


def plot_atom_ab_violin(out_path, atom_ab, title=''):
    """4-row violin grid (a, b, amp=sqrt(a^2+b^2), phase=atan2(b,a)) -- one violin per
    Q-atom id, x-axis ordered by mean amplitude (accumulate_atom_ab's
    accumulate_atom_ab already ranks/caps atom_ab that way). atom_ab: {atom_id: (a
    [n], b [n])} np.ndarray pairs."""
    ids = list(atom_ab.keys())
    a_list = [atom_ab[i][0] for i in ids]
    b_list = [atom_ab[i][1] for i in ids]
    amp_list = [np.hypot(a, b) for a, b in zip(a_list, b_list)]
    phase_list = [np.arctan2(b, a) for a, b in zip(a_list, b_list)]

    positions = np.arange(1, len(ids) + 1)
    colors = ['black'] * len(ids)

    fig, axes = plt.subplots(4, 1, figsize=(max(8, 0.35 * len(ids)), 11), sharex=True)
    rows = [('a', a_list), ('b', b_list), ('amp', amp_list), ('phase (rad)', phase_list)]
    for ax, (label, data) in zip(axes, rows):
        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.8)
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.grid(True, axis='y', alpha=0.3)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([str(i) for i in ids], rotation=90, fontsize=6)
    axes[-1].set_xlabel('Atom id (ranked by mean amplitude)')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def compute(ctx):
    ds_params = ctx.config['dataset_params']['pretrain']
    rng = random.Random(0)
    for name in ctx.args.datasets or list(ds_params):
        cfg = copy.deepcopy(ctx.config)
        subjects = cap_subjects(cfg, ds_params[name], ctx.args.max_trials, rng, min_subjects=5)
        cfg['dataset_params']['pretrain'] = {name: {**ds_params[name], 'subject_to_use': subjects}}
        ds = build_dataset_from_config(cfg, mode='pretrain')
        trials = list(range(min(ctx.args.max_trials, len(ds.base_dataset))))
        ab = accumulate_atom_ab(ctx.model, ds, trials, ctx.device, max_atoms=ctx.model.n_atoms)
        arrays = {f'a_{s}': a for s, (a, _) in ab.items()}
        arrays.update({f'b_{s}': b for s, (_, b) in ab.items()})
        save_npz(ctx.numbers('atom_distribution', f'{name}.npz'), ids=np.array(list(ab)), n_subjects=len(subjects),
                 n_windows=len(trials), **arrays)


def draw(ctx):
    ds_params = ctx.config['dataset_params']['pretrain']
    for name in ctx.args.datasets or list(ds_params):
        z = load_npz(ctx.numbers('atom_distribution', f'{name}.npz'))
        ab = {int(s): (z[f'a_{s}'], z[f'b_{s}']) for s in z['ids']}
        out = ctx.figure('atom_distribution', f'{name}.png')
        plot_atom_ab_violin(out, ab, title=f'Atom a/b/amp/phase: {name} ({int(z["n_subjects"])} subjects, '
                             f'{int(z["n_windows"])} windows)')
        print(f"  -> {out}")
