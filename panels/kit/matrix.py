"""Annotated heatmaps: a matrix with a mean and a spread written in every cell (no analysis logic)."""
import matplotlib.pyplot as plt
import numpy as np


def annotated_matrix(ax, mean, std, row_labels, col_labels, vmin, vmax, cmap, title=None, fontsize=7, outline_diag=False):
    """mean, std [R, C] (NaN cells stay blank). Every cell reads 'mean' over '±std'; the text turns white on dark cells."""
    im = ax.imshow(np.ma.masked_invalid(mean), vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=fontsize)
    mid = (vmin + vmax) / 2
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if np.isfinite(mean[i, j]):
                dark = abs(mean[i, j] - mid) > 0.35 * (vmax - vmin) if cmap in ('RdBu_r', 'RdBu') else (mean[i, j] - vmin) > 0.6 * (vmax - vmin)
                ax.text(j, i, f'{mean[i, j]:.2f}\n±{std[i, j]:.2f}', ha='center', va='center', fontsize=fontsize - 1,
                        color='white' if dark else 'black')
    if outline_diag:                                  # the diagonal is a reference (same task), not a pair of tasks
        for i in range(min(mean.shape)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec='black', lw=1.6))
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    return im


def colorbar_marks(cbar, values, color='black'):
    """Dashed reference lines across a colour bar."""
    for v in values:
        cbar.ax.axhline(v, color=color, lw=1.0, ls='--')


def matrix_row(out_path, mats, stds, titles, row_labels, col_labels, vmin, vmax, cmap, suptitle, cbar_label='', marks=(), outline_diag=False):
    """One annotated matrix per title side by side (mats, stds: lists of [R, C]) with one shared colour bar -> out_path."""
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 1.2, 3.6), squeeze=False)
    for i, (ax, m, s, t) in enumerate(zip(axes[0], mats, stds, titles)):
        im = annotated_matrix(ax, m, s, row_labels, col_labels, vmin, vmax, cmap, t, outline_diag=outline_diag)
        if i:                                         # row labels once, on the first matrix
            ax.set_yticklabels([])
    colorbar_marks(fig.colorbar(im, ax=list(axes[0]), fraction=0.015, pad=0.01, label=cbar_label), marks)
    fig.suptitle(suptitle, fontweight='bold')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
