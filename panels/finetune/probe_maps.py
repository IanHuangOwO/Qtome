"""probe_maps: where the linear probe (latent_signed head entry) reads from, per cell: virtual channel x time decision weight (event at 0 s)
and the spatial filters, over every fold and finetune seed of the cell (cells named *_seed<N> are pooled) -> probe_maps_<cell>.png.
Run it with --head <the probe's head label>, e.g. cw_probe; runs on an older trial window than the dataset's current metadata need
--event-onset DATASET=SECONDS."""
import glob
import json
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from IO.config import QTOME_OUTPUT
from IO.channels import resolve_canonical_channels
from panels.core.numbers import load_npz, save_npz
from panels.kit.channels import mirror_order
from panels.shared.head_maps import summarise, time_axis

PRESETS = set()


def add_args(flags):
    flags.add('--event-onset', action='append', default=[], metavar='DATASET=SECONDS', dest='event_onset',
              help="the event's time in the trial window, for runs made on an older trial window than the dataset's current metadata (e.g. BNCI2014001=1.0)")


def plot_probe_maps(out_path, maps, channel_names, title, event_s=None):
    """maps: {row title: (importance [K, N], spatial [K, C], t_axis [N] seconds of each patch centre)} -- each row
    has its own time axis (groups may differ in patch length / stride); the time columns share one x range.
    Left: virtual channel (ranked by importance) x time (event at 0, dashed); right: each virtual channel's
    spatial filter over electrodes."""
    fig, axes = plt.subplots(len(maps), 2, figsize=(15, 2.8 * len(maps)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.3, 1]})
    for ax_t in axes[1:, 0]:
        ax_t.sharex(axes[0, 0])
    for (ax_t, ax_s), (name, (imp, sp, t_axis)) in zip(axes, maps.items()):
        imp, sp, t_axis = np.asarray(imp), np.asarray(sp), np.asarray(t_axis)
        dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
        K = imp.shape[0]
        im = ax_t.imshow(imp, aspect='auto', cmap='viridis', origin='lower',
                         extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, -0.5, K - 0.5])
        if event_s is not None:
            ax_t.axvline(0.0, color='crimson', ls='--', lw=1.5)   # visible on the map and on the blank margin
            ax_t.text(0.0, K - 0.6, ' event', color='crimson', fontsize=8, va='top')
        ax_t.set_yticks(range(K), [f'v{k + 1}' for k in range(K)], fontsize=7)
        ax_t.set_ylabel('virtual channel\n(ranked)')
        ax_t.set_title(f'{name}: decision weight over time', fontsize=9)
        fig.colorbar(im, ax=ax_t, fraction=0.03)
        cols = mirror_order(channel_names)                      # standard mirrored channel order
        lim = np.abs(sp).max() or 1.0
        im2 = ax_s.imshow(sp[:, cols], aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower')
        ax_s.set_xticks(range(len(channel_names)), [channel_names[i] for i in cols], rotation=90, fontsize=6)
        ax_s.set_yticks(range(K), [f'v{k + 1}' for k in range(K)], fontsize=7)
        ax_s.set_title('spatial filter per virtual channel (sign-aligned)', fontsize=9)
        fig.colorbar(im2, ax=ax_s, fraction=0.03)
    axes[-1, 0].set_xlabel('time from event (s)' if event_s is not None else 'time in window (s)')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _cells(ctx):
    return sorted({re.sub(r'_seed\d+$', '', os.path.basename(d)) for bb in ctx.groups.values()
                   for d in glob.glob(f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/*')})


def compute(ctx):
    override = dict(e.split('=', 1) for e in getattr(ctx.args, 'event_onset', []))
    for cell in _cells(ctx):
        maps, cfg, notes, has_event = {}, None, set(), False
        for g, bb in ctx.groups.items():
            dirs = sorted(glob.glob(f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/{cell}')
                          + glob.glob(f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/{cell}_seed*'))
            heads = [p for d in dirs for p in sorted(glob.glob(f'{d}/finetune/run_*/head.pth'))]
            if not heads:
                continue
            try:
                imp, sp, n = summarise(heads)
            except KeyError:
                continue                                    # not a latent_signed head
            gcfg = json.load(open(f'{dirs[0]}/artifacts/config.json'))
            cfg = cfg or gcfg
            t, note = time_axis(gcfg, imp.shape[1], override)   # this group's own patch grid
            notes.add(note); has_event |= not note.startswith(' (event line omitted')
            maps[f'{g} ({n} heads)'] = (imp.numpy(), sp.numpy(), t)
        path = ctx.numbers(f'probe_maps_{cell}.npz')
        if os.path.exists(path):
            os.remove(path)
        if not maps:
            continue
        note = ''.join(sorted(n for n in notes if n))
        chans = resolve_canonical_channels(cfg['preprocess_params']['canonical_channels'])
        hc = torch.load(glob.glob(f'{QTOME_OUTPUT}/{next(iter(ctx.groups.values()))}/finetune/{ctx.head}/{cell}*/finetune/run_*/head.pth')[0],
                        map_location='cpu', weights_only=False)['head_config']
        names = [chans[i] for i in hc['channel_idx']]
        arrays = {}
        for i, (imp, sp, t) in enumerate(maps.values()):
            arrays.update({f'imp_{i}': imp, f'sp_{i}': sp, f't_{i}': t})
        save_npz(path, titles=np.array(list(maps)), names=np.array(names), note=np.array(note),
                 has_event=np.array(has_event), **arrays)


def draw(ctx):
    for cell in _cells(ctx):
        path = ctx.numbers(f'probe_maps_{cell}.npz')
        if not os.path.exists(path):
            continue
        z = load_npz(path)
        maps = {str(t): (z[f'imp_{i}'], z[f'sp_{i}'], z[f't_{i}']) for i, t in enumerate(z['titles'])}
        out = ctx.figure(f'probe_maps_{cell}.png')
        plot_probe_maps(out, maps, [str(n) for n in z['names']], f'{cell}: linear probe ({ctx.head}){str(z["note"])}',
                        event_s=0.0 if bool(z['has_event']) else None)
        print(f"  -> {out}")
