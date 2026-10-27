"""atom_seed_match: do independently trained backbones learn the same qlets? Every pair of the --group backbones (seeds of one recipe)
is matched one-to-one (Hungarian) on the phase-invariant template similarity sqrt(<D_i,D_j>^2 + <D_i,H_j>^2) (1 = the same waveform at some
phase), against the best match of random dictionaries of the same size -> atom_seed_match.png (matched similarity curves, the matching matrix,
peak frequency of matched atoms) and atom_seed_match_gallery.png (the matched waveforms of the most reproducible atoms of the reference
group, phase-aligned). Needs >= 2 groups; --head is not used. Numbers: atom_seed_match.npz."""
import itertools
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from IO.config import QLET_OUTPUT
from panels.core.context import load_model
from panels.core.numbers import load_npz, save_npz
from panels.shared.atom_dictionary import FS, atom_templates, peak_freq

PRESETS = {'standard'}
COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a5fd0')   # one per group, in order


def add_args(flags):
    flags.add('--match-threshold', type=float, default=0.9, help='atom_seed_match: a matched pair counts as the same atom above this similarity')
    flags.add('--null-draws', type=int, default=200, help='atom_seed_match: random dictionaries for the chance level')
    flags.add('--gallery', type=int, default=12, help='atom_seed_match: atoms in the waveform gallery')


def cross_similarity(a, b):
    """a, b = (D, H) [n, L] -> S[i, j] = sqrt(<D_i^a, D_j^b>^2 + <D_i^a, H_j^b>^2)."""
    return np.sqrt((a[0] @ b[0].T) ** 2 + (a[0] @ b[1].T) ** 2)


def random_dictionary(n, length, rng):
    """n random unit templates with an orthogonal unit partner each: the chance level of a match (no structure)."""
    d = rng.standard_normal((n, length)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    h = rng.standard_normal((n, length)); h -= (h * d).sum(1, keepdims=True) * d; h /= np.linalg.norm(h, axis=1, keepdims=True)
    return d, h


def match(S):
    """One-to-one assignment maximising the summed similarity -> (rows, cols, matched similarity)."""
    r, c = linear_sum_assignment(-S)
    return r, c, S[r, c]


def phase_aligned(ref_d, d, h):
    """The template d cos(t) + h sin(t) (unit norm) closest to ref_d: t = atan2(<ref, h>, <ref, d>)."""
    t = np.arctan2(ref_d @ h, ref_d @ d)
    return np.cos(t) * d + np.sin(t) * h


def aligned_topographies(groups):
    """The atom topographies of the groups' backbones (their atom_labels numbers, run the panel on each first) on the channels all of them
    share, common-average referenced and unit length -> (topo [group, atom, channel], channels, coords), or None if one backbone has no
    atom_labels numbers."""
    zs = []
    for bb in groups.values():
        path = f'{QLET_OUTPUT}/{bb}/pretrain/analysis/atom_labels.npz'
        if not os.path.exists(path):
            return None
        zs.append(load_npz(path))
    common = [c for c in zs[0]['channels'] if all(c in z['channels'] for z in zs)]
    topo = []
    for z in zs:
        t = z['topo'][:, [list(z['channels']).index(c) for c in common]]
        t = t - t.mean(1, keepdims=True)
        topo.append(t / np.linalg.norm(t, axis=1, keepdims=True))
    coords = zs[0]['coords'][[list(zs[0]['channels']).index(c) for c in common]]
    return np.stack(topo), np.array(common), coords


def compute(ctx):
    names = list(ctx.groups)
    if len(names) < 2:
        print('atom_seed_match: skipped, it needs at least two --group backbones (seeds of one recipe)')
        return
    T = {}
    for g, bb in ctx.groups.items():
        m = load_model({}, f'{QLET_OUTPUT}/{bb}/pretrain/checkpoint/last.pth', ctx.device)
        D, H, _ = atom_templates(m)
        T[g] = (D, H)
    n, length = T[names[0]][0].shape
    rng = np.random.default_rng(0)
    rand = lambda: random_dictionary(n, length, rng)
    null = np.concatenate([match(cross_similarity(T[names[0]], rand()))[2] for _ in range(ctx.args.null_draws)])
    out = dict(groups=np.array(names), null=null, D=np.stack([T[g][0] for g in names]), H=np.stack([T[g][1] for g in names]),
               freq=np.stack([peak_freq(T[g][0]) for g in names]))
    for a, b in itertools.combinations(range(len(names)), 2):
        S = cross_similarity(T[names[a]], T[names[b]])
        r, c, m = match(S)
        out[f'S_{a}_{b}'], out[f'rows_{a}_{b}'], out[f'cols_{a}_{b}'], out[f'matched_{a}_{b}'] = S, r, c, m
        print(f"  {names[a]} vs {names[b]}: matched similarity mean {m.mean():.2f}, {(m >= ctx.args.match_threshold).sum()}/{n} >= "
              f"{ctx.args.match_threshold}; chance level: mean {null.mean():.2f}, 99th percentile {np.percentile(null, 99):.2f}")
    topo = aligned_topographies(ctx.groups)
    if topo is None:
        print('  (no atom_labels numbers for every backbone: topographies not compared; run --panel atom_labels on each first)')
    else:
        out.update(topo=topo[0], topo_channels=topo[1], topo_coords=topo[2])
        for a, b in itertools.combinations(range(len(names)), 2):
            St = np.abs(topo[0][a] @ topo[0][b].T)                                     # sign-free topography similarity
            r, c = out[f'rows_{a}_{b}'], out[f'cols_{a}_{b}']
            out[f'topo_S_{a}_{b}'] = St
            other = St[~np.isin(np.arange(St.size).reshape(St.shape), r * St.shape[1] + c)]
            print(f"  {names[a]} vs {names[b]}: topography |cos| of matched atoms mean {St[r, c].mean():.2f} "
                  f"(of atoms with waveform similarity >= {ctx.args.match_threshold}: {St[r, c][out[f'matched_{a}_{b}'] >= ctx.args.match_threshold].mean():.2f}); "
                  f"unmatched pairs: mean {other.mean():.2f}, 95th percentile {np.percentile(other, 95):.2f}, {len(topo[1])} shared channels")
    out['threshold'] = np.array(ctx.args.match_threshold)
    out['gallery'] = np.array(ctx.args.gallery)
    save_npz(ctx.numbers('atom_seed_match.npz'), **out)


def draw(ctx):
    if not os.path.exists(ctx.numbers('atom_seed_match.npz')):
        return
    z = load_npz(ctx.numbers('atom_seed_match.npz'))
    names = [str(g) for g in z['groups']]
    pairs = list(itertools.combinations(range(len(names)), 2))
    thr, null = float(z['threshold']), z['null']
    n = z['D'].shape[1]
    ncol = max(len(pairs), 3); fig = plt.figure(figsize=(5.3 * ncol, 9.6)); gs = fig.add_gridspec(2, ncol)
    for k, (a, b) in enumerate(pairs):                                   # (a) the matching matrix of every pair, reordered
        axk = fig.add_subplot(gs[0, k]); S, r, c = z[f'S_{a}_{b}'], z[f'rows_{a}_{b}'], z[f'cols_{a}_{b}']
        order = np.argsort(-z[f'matched_{a}_{b}'])
        im = axk.imshow(S[np.ix_(r[order], c[order])], vmin=0, vmax=1, cmap='magma')
        axk.set_xlabel(f'{names[b]} atoms (matched order)'); axk.set_ylabel(f'{names[a]} atoms (matched order)')
        axk.set_title(f'(a{k + 1}) similarity, {names[a]} vs {names[b]}', fontsize=9, fontweight='bold'); fig.colorbar(im, ax=axk, fraction=0.046)
    ax2, ax3, ax4 = (fig.add_subplot(gs[1, k]) for k in range(3))
    ax2.axhspan(0, np.percentile(null, 99), color='#cccccc', alpha=0.6, label='random dictionary (99th percentile)')
    ax2.axhline(thr, color='k', ls='--', lw=0.8, label=f'same atom ({thr:g})')
    for k, (a, b) in enumerate(pairs):                                   # (b) sorted matched similarity per pair
        m = np.sort(z[f'matched_{a}_{b}'])[::-1]
        ax2.plot(range(1, n + 1), m, 'o-', ms=3, color=COLORS[k % len(COLORS)],
                 label=f'{names[a]} vs {names[b]}: {(m >= thr).sum()}/{n} >= {thr:g}')
    ax2.set_xlabel('atom (most to least similar match)'); ax2.set_ylabel('matched similarity'); ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=7, loc='lower left')
    ax2.set_title('(b) every atom against its one-to-one match', fontsize=9, fontweight='bold')
    for k, (a, b) in enumerate(pairs):                                   # (c) peak frequency of matched atoms
        r, c = z[f'rows_{a}_{b}'], z[f'cols_{a}_{b}']
        ax3.scatter(z['freq'][a][r], z['freq'][b][c], s=18, color=COLORS[k % len(COLORS)], alpha=0.8,
                    label=f'{names[a]} vs {names[b]}')
    lim = max(float(z['freq'].max()), 1.0) * 1.05
    ax3.plot([0, lim], [0, lim], 'k:', lw=0.8)
    ax3.set_xlabel('peak frequency of an atom (Hz)'); ax3.set_ylabel('peak frequency of its match (Hz)')
    ax3.set_title('(c) matched atoms sit at the same frequency', fontsize=9, fontweight='bold'); ax3.legend(fontsize=7)
    M = np.full((len(names), len(names)), np.nan)                          # (d) every seed against every other: atoms matched at >= the threshold
    for a, b in pairs:
        M[a, b] = M[b, a] = (z[f'matched_{a}_{b}'] >= thr).mean()
    np.fill_diagonal(M, 1.0)
    im = ax4.imshow(M, vmin=0, vmax=1, cmap='magma'); ax4.set_xticks(range(len(names))); ax4.set_xticklabels(names); ax4.set_yticks(range(len(names))); ax4.set_yticklabels(names)
    for a in range(len(names)):
        for b in range(len(names)):
            ax4.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', color='white' if M[a, b] < 0.6 else 'black', fontsize=10)
    ax4.set_title(f'(d) share of atoms matched at >= {thr:g}, every seed pair', fontsize=9, fontweight='bold'); fig.colorbar(im, ax=ax4, fraction=0.046)
    fig.suptitle('Do independently trained backbones learn the same qlets?', fontweight='bold')
    out = ctx.figure('atom_seed_match.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f'  -> {out}')

    # gallery: the reference group's atoms matched best in every other group, waveforms phase-aligned to the reference
    others = [g for g in range(1, len(names))]
    ref_match = {}
    for g in others:
        r, c = z[f'rows_0_{g}'], z[f'cols_0_{g}']
        ref_match[g] = (dict(zip(r.tolist(), c.tolist())), dict(zip(r.tolist(), z[f'matched_0_{g}'].tolist())))
    worst = np.array([min(ref_match[g][1][i] for g in others) for i in range(n)])
    top = np.argsort(-worst)[:int(z['gallery'])]
    cols = 4
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 1.9 * rows), squeeze=False, sharex=True)
    t = np.arange(z['D'].shape[2]) / FS * 1000
    for ax, i in zip(axes.ravel(), top):
        ref = z['D'][0][i]
        ax.plot(t, ref, color=COLORS[0], lw=1.6, label=names[0])
        for g in others:
            j = ref_match[g][0][i]
            ax.plot(t, phase_aligned(ref, z['D'][g][j], z['H'][g][j]), color=COLORS[g % len(COLORS)], lw=1.1, label=names[g])
        ax.set_title(f'atom {i}: {z["freq"][0][i]:.0f} Hz, worst match {worst[i]:.2f}', fontsize=8)
        ax.tick_params(labelsize=7); ax.set_yticks([])
    for ax in axes.ravel()[len(top):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('ms', fontsize=7)
    axes[0, 0].legend(fontsize=6, loc='lower left')
    fig.suptitle(f'Most reproducible atoms of {names[0]} and their matches in the other backbones (phase-aligned templates)', fontweight='bold')
    out = ctx.figure('atom_seed_match_gallery.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f'  -> {out}')
    if 'topo' in z:
        draw_topography(ctx, z, names, pairs, top[:min(8, len(top))])


def draw_topography(ctx, z, names, pairs, order_atoms):
    """atom_seed_match_topography.png: waveform similarity against topography similarity of every matched atom, and the topographies of the
    most reproducible atoms in every group. Needs the atom_labels numbers of every backbone."""
    from panels.kit.topomap import build_triangulation, draw_topomap, project_coords_2d
    thr = float(z['threshold'])
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax = fig.add_subplot(gs[0])
    allnull = np.concatenate([z[f'topo_S_{a}_{b}'][~np.isin(np.arange(z[f'topo_S_{a}_{b}'].size).reshape(z[f'topo_S_{a}_{b}'].shape),
                              z[f'rows_{a}_{b}'] * z[f'topo_S_{a}_{b}'].shape[1] + z[f'cols_{a}_{b}'])] for a, b in pairs])
    ax.axhspan(0, float(np.mean(allnull)), color='#cccccc', alpha=0.6, label=f'unmatched atom pairs: mean {np.mean(allnull):.2f}')
    ax.axhline(np.percentile(allnull, 95), color='#888888', ls=':', lw=1, label=f'unmatched pairs, 95th percentile {np.percentile(allnull, 95):.2f}')
    ax.axvline(thr, color='k', ls='--', lw=0.8)
    for k, (a, b) in enumerate(pairs):
        r, c = z[f'rows_{a}_{b}'], z[f'cols_{a}_{b}']
        ax.scatter(z[f'matched_{a}_{b}'], z[f'topo_S_{a}_{b}'][r, c], s=22, color=COLORS[k % len(COLORS)], alpha=0.85, label=f'{names[a]} vs {names[b]}')
    ax.set_xlabel('waveform similarity of the matched atoms'); ax.set_ylabel('topography similarity (|cosine|, sign-free)')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, loc='lower right')
    hit = []
    for a, b in pairs:                                                    # is the matched atom also the closest topography of the other backbone?
        St, good = z[f'topo_S_{a}_{b}'], z[f'matched_{a}_{b}'] >= thr
        rank = np.array([int((St[i] > St[i, j]).sum()) + 1 for i, j in zip(z[f'rows_{a}_{b}'], z[f'cols_{a}_{b}'])])
        hit.append(f'{names[a]}-{names[b]} {int((rank[good] == 1).sum())}/{int(good.sum())}')
    ax.set_title('Do atoms with the same waveform sit at the same place on the scalp?\n'
                 f'the matched atom is also the closest topography (chance 1/{z["D"].shape[1]}): ' + ', '.join(hit), fontsize=8, fontweight='bold')
    pos2d = project_coords_2d(z['topo_coords']); tri = build_triangulation(pos2d)
    G, top = len(names), order_atoms
    sub = gs[1].subgridspec(len(top), G, hspace=0.05, wspace=0.05)
    for row, i in enumerate(top):
        idx = {0: i}
        for g in range(1, G):
            idx[g] = int(z[f'cols_0_{g}'][list(z[f'rows_0_{g}']).index(i)])
        for g in range(G):
            axt = fig.add_subplot(sub[row, g]); v = z['topo'][g][idx[g]]
            if g and float(z['topo'][0][i] @ v) < 0:
                v = -v                                                    # sign for display only: as the reference atom
            draw_topomap(axt, pos2d, v, cmap='RdBu_r', vmin=-np.abs(v).max(), vmax=np.abs(v).max(), triang=tri)
            if row == 0:
                axt.set_title(names[g], fontsize=8)
            if g == 0:
                axt.set_ylabel(f'atom {i}\n{z["freq"][0][i]:.0f} Hz', rotation=0, ha='right', va='center', fontsize=7)
            else:
                axt.text(0.5, -0.02, f'|cos| {abs(float(z["topo"][0][i] @ z["topo"][g][idx[g]])):.2f}', transform=axt.transAxes, ha='center', va='top', fontsize=6)
    fig.suptitle(f'Atom topographies across backbones ({len(z["topo_channels"])} shared channels, common-average reference)', fontweight='bold')
    out = ctx.figure('atom_seed_match_topography.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'  -> {out}')
