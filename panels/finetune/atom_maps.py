"""atom_maps: where a head with a atom_power entry reads from, per cell and group (a latent_signed entry of the same head, if any, is drawn
on top: virtual channel x time and spatial filters, as probe_maps; a signed_ab entry is drawn on top in Q-atom space instead -- Q-atom x
time, gain-scaled, scalp maps of its top Q-atoms): each Q-atom's learned time weights (Q-atoms ranked by decision importance, labelled
with their template's peak frequency) and, for the most important Q-atoms, which electrodes' power the decision uses, over every fold and
finetune seed of the cell -> atom_maps_<cell>_<group>.png. Run it with --head <a label whose head has atom_power>, e.g. combined or
cw_atom. One figure per group: Q-atoms are each backbone's own dictionary, not comparable across."""
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from IO.channels import resolve_canonical_channels
from IO.channels import get_standard_coords
from model.factory import build_from_checkpoint
from model.Qtome.finetune.feature_cache import CachedAtomDataset, get_atom_cache
from panels.core.numbers import load_npz, save_npz
from panels.kit.channels import mirror_order
from panels.shared.finetune_runs import cell_names, cell_run_dirs, head_files
from panels.shared.head_maps import summarise, summarise_atoms, summarise_signed_atoms, time_axis

PRESETS = set()


def add_args(flags):
    flags.add('--event-onset', action='append', default=[], metavar='DATASET=SECONDS', dest='event_onset',
              help="the event's time in the trial window, for runs made on an older trial window than the dataset's current metadata (e.g. BNCI2014001=1.0)")


def _atom_block(fig, gs, r0, tw, imp, chan, t_axis, atom_labels, xy, channel_names, event_s, title, cbar_label,
                 top=8, ncol=4):
    """Two grid rows from r0: Q-atom x time map (rows ranked by importance) and scalp maps of the top Q-atoms."""
    order = np.argsort(-imp)
    share = imp / imp.sum()
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
    ax = fig.add_subplot(gs[r0:r0 + 2, 0])
    im = ax.imshow(tw[order], aspect='auto', cmap='viridis', origin='upper',
                   extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, len(order) - 0.5, -0.5])
    ax.set_yticks(range(len(order)), [f'{atom_labels[s]} ({share[s]:.0%})' for s in order], fontsize=7)
    if event_s is not None:
        ax.axvline(0.0, color='crimson', ls='--', lw=1.5)
    ax.set_xlabel('time from event (s)' if event_s is not None else 'time in window (s)')
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, label=cbar_label)
    shown = order[:min(top, 2 * ncol)]
    vmax = max(float(chan[shown].max()), 1e-12)                  # one colour scale for every scalp map
    for i, s in enumerate(shown):
        a = fig.add_subplot(gs[r0 + i // ncol, 1 + i % ncol])
        sc = a.scatter(xy[:, 0], xy[:, 1], c=chan[s], cmap='Reds', vmin=0, vmax=vmax, s=90,
                       edgecolors='k', linewidths=0.3)
        if len(channel_names) <= 8:
            for (x, y), n in zip(xy, channel_names):
                a.annotate(n, (x, y), fontsize=6, ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        pad = 0.02
        a.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); a.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        a.set_title(f'{atom_labels[s]} ({share[s]:.0%})', fontsize=8)
    fig.colorbar(sc, cax=fig.add_subplot(gs[r0:r0 + 2, -1]), label='channel weight (per head, each atom max = 1)')


def plot_atom_maps(out_path, tw, imp, chan, t_axis, atom_labels, xy, channel_names, title, event_s=None, top=8,
                    z_maps=None, signed_maps=None):
    """Q-atom-power half of a head (bottom): per-atom time weights (rows = Q-atoms ranked by importance, label = Q-atom,
    template peak and importance share) and the channel map of the `top` most important Q-atoms on the scalp.
    The same head's signed half, if any, on top on the same time axis:
    z_maps (importance [K, N], spatial [K, C]) of a latent_signed entry -- virtual channel x time and spatial filters
    (z has no Q-atom axis); signed_maps (importance [S, N], per-atom importance [S], channel map [S, C]) of a signed_ab
    entry -- drawn in Q-atom space like the power half, so the two halves read side by side."""
    tw, imp, chan, t_axis, xy = (np.asarray(v) for v in (tw, imp, chan, t_axis, xy))
    ncol = 4
    if signed_maps is not None:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2 + ncol, width_ratios=[2.2] + [1] * ncol + [0.08])
        stw, simp, schan = (np.asarray(v) for v in signed_maps)
        _atom_block(fig, gs, 0, stw, simp, schan, t_axis, atom_labels, xy, channel_names, event_s,
                     'signed atom half (signed_ab): decision weight per atom x time (gain-scaled), ranked',
                     'importance share', top, ncol)
        _atom_block(fig, gs, 2, tw, imp, chan, t_axis, atom_labels, xy, channel_names, event_s,
                     'atom power half (atom_power): time weight per atom, ranked by importance', 'time weight',
                     top, ncol)
        fig.suptitle(title, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return
    order = np.argsort(-imp)
    share = imp / imp.sum()
    z0 = 0 if z_maps is None else 1
    fig = plt.figure(figsize=(16, 6 + 2.6 * z0))
    gs = fig.add_gridspec(2 + z0, 2 + ncol, width_ratios=[2.2] + [1] * ncol + [0.08], height_ratios=[1.1] * z0 + [1, 1])
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
    if z_maps is not None:
        zi, zs = (np.asarray(v) for v in z_maps)
        K = zi.shape[0]
        az = fig.add_subplot(gs[0, 0])
        zshare = zi.sum(1) / zi.sum()                            # rows come ranked (v1 = most important)
        vlab = [f'v{k + 1} ({zshare[k]:.0%})' for k in range(K)]
        imz = az.imshow(zi, aspect='auto', cmap='viridis', origin='upper',
                        extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, K - 0.5, -0.5])
        az.set_yticks(range(K), vlab, fontsize=7)
        az.set_title('z half (latent_signed): decision weight per virtual channel', fontsize=9)
        if event_s is not None:
            az.axvline(0.0, color='crimson', ls='--', lw=1.5)
        fig.colorbar(imz, ax=az, fraction=0.03)
        asp = fig.add_subplot(gs[0, 1:])
        lim = np.abs(zs).max() or 1.0
        cols = mirror_order(channel_names)
        ims = asp.imshow(zs[:, cols], aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='upper')
        asp.set_xticks(range(len(channel_names)), [channel_names[i] for i in cols], rotation=90, fontsize=6)
        asp.set_yticks(range(K), vlab, fontsize=7)
        asp.set_title('z half: spatial filter per virtual channel (sign-aligned)', fontsize=9)
        fig.colorbar(ims, ax=asp, fraction=0.03)
    _atom_block(fig, gs, z0, tw, imp, chan, t_axis, atom_labels, xy, channel_names, event_s,
                 'atom half (atom_power): time weight per atom, ranked by importance', None, top, ncol)
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _atom_labels(backbone_ckpt, sample_freq):
    """'s<i> <peak> Hz' per Q-atom, from the backbone's unit templates."""
    atoms = build_from_checkpoint(torch.load(backbone_ckpt, map_location='cpu', weights_only=False)).atoms
    with torch.no_grad():
        D, _ = atoms.templates()
    f = torch.fft.rfftfreq(D.shape[-1], 1.0 / sample_freq)
    peak = f[torch.fft.rfft(D.float(), dim=-1).abs().argmax(-1)]
    return [f's{i} {float(p):.0f} Hz' for i, p in enumerate(peak)]


def compute(ctx):
    override = dict(e.split('=', 1) for e in getattr(ctx.args, 'event_onset', []))
    for g, bb in ctx.groups.items():
        for old in glob.glob(ctx.numbers(f'atom_maps_*_{g}.npz')):
            os.remove(old)
        for cell in cell_names([bb], ctx.head):
            dirs = cell_run_dirs(bb, ctx.head, cell)
            heads = head_files(dirs)
            if not heads:
                continue
            try:
                tw, imp, chan, n = summarise_atoms(heads)
            except KeyError:
                continue                                    # no atom_power entry in this head
            cfg = json.load(open(f'{dirs[0]}/artifacts/config.json'))
            arrays = {}
            try:
                z_imp, z_sp, _ = summarise(heads)             # the same head's latent_signed half, if it has one
                arrays.update(z_imp=z_imp.numpy(), z_sp=z_sp.numpy())
            except KeyError:
                pass
            if any(f['type'] == 'signed_ab' for f in torch.load(heads[0], map_location='cpu', weights_only=False)['head_config']['features']):
                ds = next(iter(cfg['dataset_params']['finetune']))   # signed_ab half in Q-atom space, gain-scaled
                subs = list(json.load(open(cfg['dataset_params']['finetune'][ds]['dataset_path'] + '/metadata.json'))['data_structure'])
                amp = CachedAtomDataset(get_atom_cache(cfg, ds, subs), subs).amp
                s_tw, s_imp, s_chan, _ = summarise_signed_atoms(heads, amp)
                arrays.update(s_tw=s_tw.numpy(), s_imp=s_imp.numpy(), s_chan=s_chan.numpy())
            t, note = time_axis(cfg, tw.shape[1], override)
            h0 = torch.load(heads[0], map_location='cpu', weights_only=False)
            chans = resolve_canonical_channels(cfg['preprocess_params']['canonical_channels'])
            names = [chans[i] for i in h0['head_config']['channel_idx']]
            xy = np.array([get_standard_coords(c)[:2] for c in names])
            labels = _atom_labels(h0['backbone_checkpoint'], float(cfg['preprocess_params']['sample_freq']))
            save_npz(ctx.numbers(f'atom_maps_{cell}_{g}.npz'), tw=tw.numpy(), imp=imp.numpy(), chan=chan.numpy(), t=t,
                     labels=np.array(labels), xy=xy, names=np.array(names),
                     title=np.array(f'{cell}, {g} ({bb}): {ctx.head}, {n} heads{note}'),
                     has_event=np.array(not note.startswith(' (event line omitted')), **arrays)


def draw(ctx):
    for g in ctx.groups:
        for path in sorted(glob.glob(ctx.numbers(f'atom_maps_*_{g}.npz'))):
            z = load_npz(path)
            out = ctx.figure(os.path.basename(path)[:-len('.npz')] + '.png')
            plot_atom_maps(out, z['tw'], z['imp'], z['chan'], z['t'], [str(x) for x in z['labels']], z['xy'],
                           [str(x) for x in z['names']], str(z['title']),
                           event_s=0.0 if bool(z['has_event']) else None,
                           z_maps=(z['z_imp'], z['z_sp']) if 'z_imp' in z else None,
                           signed_maps=(z['s_tw'], z['s_imp'], z['s_chan']) if 's_tw' in z else None)
            print(f"  -> {out}")
