"""time_weights: where in the trial a learned time pool reads from. Per cell, one Q-atom x patch heatmap per group and head entry,
averaged over the cell's fold/subject heads (each trains its own) -> time_weights_<cell>.png, event onset marked."""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from IO.config import QTOME_OUTPUT
from IO.event_onset import event_onset_patch
from panels.core.numbers import load_npz, save_npz

PRESETS = {'standard'}

def time_pool_weights(head_pth):
    """A finetune head's learned time-pool weights, w[s, n] = softmax_n(sum_r p[r, s] q[r, n]) per head
    entry with time_pool 'learned' -> {entry name: [S, N] numpy}."""
    sd = torch.load(head_pth, map_location='cpu', weights_only=False)['model_state_dict']
    return {k[:-len('.time.p')].removeprefix('entries.'):
            torch.softmax(torch.einsum('rs,rn->sn', sd[k], sd[k[:-1] + 'q']), -1).numpy()
            for k in sd if k.endswith('.time.p')}


def plot_time_weights(out_path, maps, title, event_patch=None):
    """One Q-atom x patch heatmap per row; maps: {row title: [S, N] weights}."""
    fig, axes = plt.subplots(len(maps), 1, figsize=(8, 2.2 * len(maps)), squeeze=False)
    for ax, (name, w) in zip(axes[:, 0], maps.items()):
        im = ax.imshow(w, aspect='auto', cmap='viridis', origin='lower')
        if event_patch is not None:
            ax.axvline(event_patch, color='w', ls='--', lw=1.2)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel('atom')
        fig.colorbar(im, ax=ax, fraction=0.03)
    axes[-1, 0].set_xlabel('patch (time)' + ('; dashed = event onset' if event_patch is not None else ''))
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _cells(ctx):
    return sorted({os.path.basename(d) for bb in ctx.groups.values()
                   for d in glob.glob(f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/*')})


def compute(ctx):
    for cell in _cells(ctx):
        maps, cfg = {}, None
        for g, bb in ctx.groups.items():
            run_dir = f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/{cell}'
            heads = [time_pool_weights(p) for p in sorted(glob.glob(f'{run_dir}/finetune/run_*/head.pth'))]
            for entry in sorted({e for w in heads for e in w}):
                ws = [w[entry] for w in heads if entry in w]
                ws = [w for w in ws if w.shape == ws[0].shape]
                maps[f'{g} / {entry} (mean of {len(ws)} heads)'] = np.mean(ws, 0)
            if cfg is None and os.path.exists(f'{run_dir}/artifacts/config.json'):
                cfg = json.load(open(f'{run_dir}/artifacts/config.json'))
        if not maps:
            continue
        ev = event_onset_patch(cfg, next(iter(cfg['dataset_params']['finetune']))) if cfg else None
        save_npz(ctx.numbers(f'time_weights_{cell}.npz'), titles=np.array(list(maps)), event=np.array(np.nan if ev is None else ev),
                 **{f'm{i}': w for i, w in enumerate(maps.values())})


def draw(ctx):
    for cell in _cells(ctx):
        path = ctx.numbers(f'time_weights_{cell}.npz')
        if not os.path.exists(path):
            continue
        z = load_npz(path)
        maps = {str(t): z[f'm{i}'] for i, t in enumerate(z['titles'])}
        ev = None if np.isnan(z['event']) else z['event'].item()
        out = ctx.figure(f'time_weights_{cell}.png')
        plot_time_weights(out, maps, f'{cell}: learned time-pool weights ({ctx.head})', ev)
        print(f"  -> {out}")
