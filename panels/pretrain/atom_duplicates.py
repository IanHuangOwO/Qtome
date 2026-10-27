"""atom_duplicates: are qlets copies of each other up to phase? Prints pairs above --dup-threshold (default 0.9) and each
Qlet's nearest neighbour -> analysis/pretrain/atom_duplicates.png."""
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz, save_npz
from panels.shared.atom_dictionary import atom_similarity

PRESETS = {'standard', 'quick'}


def add_args(flags):
    flags.add('--dup-threshold', type=float, default=0.9, help='atom_duplicates: pair threshold')


def plot_similarity_matrix(out_path, sim, labels, title):
    """Square [0, 1] similarity heatmap with tick labels."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * n), max(5, 0.3 * n)))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap='magma')
    ax.set_xticks(range(n), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n), labels, fontsize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def compute(ctx):
    sim, label = atom_similarity(ctx.model)
    save_npz(ctx.numbers('atom_duplicates.npz'), sim=sim, labels=np.array(label))


def draw(ctx):
    z = load_npz(ctx.numbers('atom_duplicates.npz'))
    sim, label = z['sim'], [str(x) for x in z['labels']]
    thr = ctx.args.dup_threshold
    pairs = sorted(((sim[i, j], i, j) for i, j in zip(*np.triu_indices(len(label), 1)) if sim[i, j] >= thr),
                   reverse=True)
    print(f"{len(label)} alive atoms; {len(pairs)} pair(s) with similarity >= {thr}")
    for s, i, j in pairs:
        print(f"  {s:.3f}  #{label[i]} <-> #{label[j]}")
    nn = np.nanargmax(sim, axis=1)
    print("nearest neighbour: " + ', '.join(f"#{label[k]}->#{label[j]} {sim[k, j]:.2f}" for k, j in enumerate(nn)))
    out = ctx.figure('atom_duplicates.png')
    plot_similarity_matrix(out, sim, label, f'Phase-invariant atom similarity sqrt(<Di,Dj>^2+<Di,Hj>^2)\n'
                                            f'{len(pairs)} pair(s) >= {thr}')
    print(f"  -> {out}")
