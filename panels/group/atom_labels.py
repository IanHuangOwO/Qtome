"""atom_labels (group): are the ICLabel classes of the qlets the same from seed to seed? Per recipe: the class counts of each seed (stacked bars) and, for atoms
matched across seeds by template similarity (>= --label-match, Hungarian, phase-invariant), the share whose top class
agrees, against the share when the partners are shuffled -> atom_labels.png and atom_labels.md. Needs the pretrain panel atom_labels on each backbone."""
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from IO.config import QLET_OUTPUT
from panels.core.context import load_model
from panels.core.numbers import load_npz
from panels.shared.atom_dictionary import atom_templates
from panels.shared.groups import numbers_path, recipes

PRESETS = {'standard'}
CLASSES = ('brain', 'muscle', 'eye', 'heart', 'line noise', 'channel noise', 'other')
COLORS = ('#1baf7a', '#eb6834', '#2a78d6', '#9a5fd0', '#eda100', '#888888', '#cccccc')


def add_args(flags):
    flags.add('--label-match', type=float, default=0.8, help='atom_labels (group): template similarity above which a matched pair counts')


def similarity(a, b):
    n = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    (da, ha), (db_, hb) = (tuple(map(n, a)), tuple(map(n, b)))
    return np.sqrt((da @ db_.T) ** 2 + (da @ hb.T) ** 2)


def compute(ctx):
    rng = np.random.default_rng(0); out, lines = {}, []
    for r, members in recipes(ctx.groups).items():
        Z, T = [], []
        for g, bb in members:
            p = numbers_path(bb, 'atom_labels.npz')
            try:
                z = load_npz(p)
            except FileNotFoundError:
                print(f'  {g}: no atom_labels.npz; left out'); continue
            m = load_model({}, f'{QLET_OUTPUT}/{bb}/pretrain/checkpoint/last.pth', ctx.device)
            D, H, _ = atom_templates(m)
            Z.append(z); T.append((np.asarray(D), np.asarray(H)))
        if not Z:
            continue
        top = [z['probs'].argmax(1) for z in Z]
        out[f'{r}__counts'] = np.array([[(t == c).sum() for c in range(len(CLASSES))] for t in top])
        agree, chance, nmatch = [], [], 0
        for i, j in itertools.combinations(range(len(Z)), 2):
            S = similarity(T[i], T[j]); rr, cc = linear_sum_assignment(-S); good = S[rr, cc] > ctx.args.label_match
            agree += list(top[i][rr][good] == top[j][cc][good]); nmatch += int(good.sum())
            chance += [np.mean(top[i][rr][good] == rng.permutation(top[j][cc])[good]) for _ in range(100)]
        out[f'{r}__agree'] = np.array([np.mean(agree) if agree else np.nan, np.mean(chance) if chance else np.nan, nmatch, len(Z)])
        lines.append(f'| {r} | {len(Z)} | ' + ' | '.join(f'{out[f"{r}__counts"][:, c].mean():.1f}' for c in range(len(CLASSES))) +
                     f' | {out[f"{r}__agree"][0]:.2f} (chance {out[f"{r}__agree"][1]:.2f}, {nmatch} matched atoms) |')
    if not out:
        print('  no atom_labels numbers: run the pretrain panel atom_labels on each backbone'); return
    np.savez(ctx.numbers('atom_labels_group.npz'), recipes=np.array(list(dict.fromkeys(k.split('__')[0] for k in out))), **out)
    head = ['| recipe | seeds | ' + ' | '.join(CLASSES) + ' | same top class in matched atoms |', '|---|---|' + '---|' * (len(CLASSES) + 1)]
    open(ctx.numbers('atom_labels.md'), 'w').write('\n'.join(head + lines) + '\n\nMean atoms per ICLabel class over seeds; agreement of the top class between seeds for atoms matched by waveform.\n')
    print('\n'.join(head + lines))


def draw(ctx):
    z = np.load(ctx.numbers('atom_labels_group.npz')); recs = [str(r) for r in z['recipes']]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.6, 1]})
    x, ticks, labels = 0, [], []
    for r in recs:
        c = z[f'{r}__counts']
        for s in range(len(c)):
            bottom = 0
            for k in range(len(CLASSES)):
                ax[0].bar(x, c[s, k], bottom=bottom, color=COLORS[k], label=CLASSES[k] if (x == 0) else None); bottom += c[s, k]
            ticks.append(x); labels.append(f'{r}\ns{s + 1}'); x += 1
        x += 0.6
    ax[0].set_xticks(ticks); ax[0].set_xticklabels(labels, fontsize=8); ax[0].set_ylabel('atoms'); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title('ICLabel class of each atom (top class), per seed', fontsize=10, fontweight='bold')
    a = np.array([z[f'{r}__agree'] for r in recs]); xs = np.arange(len(recs))
    ax[1].bar(xs - .2, a[:, 0], .38, color='#2a78d6', label='matched atoms'); ax[1].bar(xs + .2, a[:, 1], .38, color='#cccccc', label='shuffled partners')
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f'{r}\n({int(n)} atoms)' for r, n in zip(recs, a[:, 2])], fontsize=8); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].set_ylabel('same top class in another seed'); ax[1].set_title('Label consistency across seeds', fontsize=10, fontweight='bold')
    fig.tight_layout(); out = ctx.figure('atom_labels.png'); fig.savefig(out, dpi=130); plt.close(fig); print(f'  -> {out}')
