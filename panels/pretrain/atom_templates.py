"""atom_templates: every Q-atom's template D and quadrature partner H -> analysis/pretrain/atom_templates.png."""
import matplotlib.pyplot as plt
import numpy as np

from panels.core.numbers import load_npz, save_npz
from panels.shared.atom_dictionary import atom_templates

PRESETS = {'standard', 'quick'}


def plot_atom_templates(out_path, D, H, ids):
    """Grid of Q-atom templates: D solid, its quadrature partner H dashed."""
    ncols = 8
    nrows = (len(ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.6 * nrows), squeeze=False)
    t = np.arange(D.shape[1])
    for ax, sid in zip(axes.flat, ids):
        ax.plot(t, D[sid], color='steelblue', lw=1.2, label='D')
        ax.plot(t, H[sid], color='crimson', lw=1.0, ls='--', label='H')
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'#{sid}', fontsize=7)
    for ax in axes.flat[len(ids):]:
        ax.axis('off')
    axes[0][0].legend(fontsize=6, loc='upper right')
    fig.suptitle(f'Atom templates (D solid, H dashed): {len(ids)}/{D.shape[0]} alive', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def compute(ctx):
    D, H, ids = atom_templates(ctx.model)
    save_npz(ctx.numbers('atom_templates.npz'), D=D, H=H, ids=ids)


def draw(ctx):
    z = load_npz(ctx.numbers('atom_templates.npz'))
    out = ctx.figure('atom_templates.png')
    plot_atom_templates(out, z['D'], z['H'], z['ids'])
    print(f"  -> {out}")
