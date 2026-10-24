"""The whole-trial Q-atom gallery figure (snapshot and class_snapshots): raw / reconstruction PSD on top, then per Q-atom its
topography, PSD, phase and waveform."""
import math

import matplotlib.pyplot as plt
import numpy as np

from panels.kit.topomap import build_triangulation, draw_topomap


def _log_pow(x):
    """log1p of a nonnegative power/PSD quantity (topo=L2 norm, psd=real^2+imag^2 — both
    always >=0, np.maximum guards float rounding noise below 0). log1p(0)=0 exactly, so
    the zero-fill "unused" floor (see plot_atom_by_patch/_cell's vmin=0 anchoring)
    survives the log transform unchanged, while still compressing the large dynamic range
    a handful of high-power channels/bins would otherwise dominate on a linear scale."""
    return np.log1p(np.maximum(x, 0.0))


def _log_signed(x):
    """Signed counterpart of _log_pow for quantities where polarity matters (a Q-atom's
    per-channel amp — its mixing/topomap column): sign(x)*log1p(|x|). Odd function, 0
    maps to exactly 0, magnitudes compress the same way _log_pow's do, sign survives —
    so a symmetric diverging color scale centered on 0 stays meaningful after the
    transform."""
    return np.sign(x) * np.log1p(np.abs(x))


def plot_atom_gallery(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                        psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                        phase_ch_x=None, waveforms=None,
                        subject_id=None, trial_idx=None, epoch_tag='',
                        unit_label='Atom', n_per_row=5):
    """
    Whole-trial panel: the trial-wide Raw/Full-Recon view and every Q-atom's trial summary.

    Header row: Raw and Full-Recon (whole trial) topo + PSD, one block each. Below: a
    grid of `n_per_row` blocks per row, one block per Q-atom (see
    viz.extract.extract_atom_gallery), sorted by accumulated importance descending,
    reading left-to-right then top-to-bottom. Each block: topo on the LEFT (spanning
    the block's full height), PSD (channel x freq, from psd_x) top-right, and — when
    phase_ch_x is given — a per-channel PHASE bar chart bottom-right, directly under
    the PSD and sharing its channel-axis orientation (see _cell) so a phase shift
    across channels reads paired against that same channel's PSD row. A single
    vertical bar chart at the right edge spans just the Q-atom grid (not the Raw/Recon
    header), one horizontal bar per Q-atom in the SAME sorted order (top = highest
    importance) — paired 1:1 with the gallery (by rank, not by grid position, since
    the grid wraps n_per_row-wide) so "how much did this Q-atom's real content matter"
    reads directly alongside "what did it actually look like".

    psd_ch_x: [C, Q]. psd_x: [Q, C, F]. phase_ch_x: [C, Q] radians or None (Raw/Recon
    header cells never show phase — there's no atom/quadrature structure to a raw
    signal — and passing None here entirely skips the phase row for Q-atom cells too).
    importance: [Q], Q-atom q = column q. All topo/PSD cells are log1p-scaled (see _log_pow), same reasoning as
    plot_atom_by_patch.

    waveforms: optional list of Q 1-D arrays of the real trial length (see
    viz.extract.extract_atom_gallery): the Q-atom's decoded content at ONE pinned channel over
    the whole trial, rendered as an extra full-block-width row under each Q-atom's PSD/phase pair.

    """
    Q = psd_ch_x.shape[1]
    order = np.argsort(-importance)
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    triang = build_triangulation(pos2d)

    # Each block is 2 grid-columns wide (topo | psd-over-phase) instead of 1 — topo
    # spans both of the block's rows on the left, PSD/phase stack on the right, so
    # topo_w/right_w need to roughly sum-match (psd_h+phase_h) for topo to land near
    # square (draw_topomap forces equal aspect, same col_w-matching rationale as
    # plot_atom_panel).
    topo_w, right_w, bar_w = 2.6, 2.6, 3.5
    psd_h, phase_h = 1.4, 1.3
    wave_h = 0.8   # waveform row height (only present when waveforms given)
    has_wave = waveforms is not None
    rows_per_block = 2 + (1 if has_wave else 0)
    block_h = psd_h + phase_h + (wave_h if has_wave else 0)
    n_per_row = max(2, n_per_row)  # header needs 2 blocks (Raw, Full Recon) side by side
    n_atom_rows = math.ceil(Q / n_per_row) if Q else 0
    n_cols = n_per_row * 2
    header_rows = 2  # psd-height row, phase-height row (topo spans both)
    total_rows = header_rows + n_atom_rows * rows_per_block
    suptitle_in, margin_in = 1.6, 0.15
    fig_w = (topo_w + right_w) * n_per_row + bar_w
    fig_h = (psd_h + phase_h) + block_h * n_atom_rows + suptitle_in + margin_in
    block_ratios = [psd_h, phase_h] + ([wave_h] if has_wave else [])
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(total_rows, n_cols + 1,
                           height_ratios=[psd_h, phase_h] + block_ratios * n_atom_rows,
                           width_ratios=[topo_w, right_w] * n_per_row + [bar_w],
                           left=0.02, right=0.98, top=1 - suptitle_in / fig_h, bottom=margin_in / fig_h,
                           hspace=0.6, wspace=0.35)
    fig.suptitle(f"Whole-Trial Raw / Recon / {unit_label} Gallery "
                 f"({unit_label}s sorted by accumulated importance) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    def _cell(topo_row, block_col, power, psd_cf, phase_c, label, color, signed=False):
        # signed: Q-atom topos are the SIGNED trial-mean per-channel amp (the mixing/
        # topomap column, see extract_atom_gallery) — diverging RdBu_r, symmetric
        # limits, 0 = white, so dipole polarity reads directly (same rationale as
        # plot_atom_by_patch's Q-atom cells). Raw/Recon header stays unsigned power.
        topo_col, right_col = block_col * 2, block_col * 2 + 1
        psd_cf = _log_pow(psd_cf)
        ax_topo = fig.add_subplot(gs[topo_row:topo_row + 2, topo_col])
        if signed:
            v = _log_signed(power)
            vlim = max(np.abs(v).max(), 1e-12)
            im_t = draw_topomap(ax_topo, pos2d, v, cmap='RdBu_r',
                                 vmin=-vlim, vmax=vlim, triang=triang)
        else:
            v = _log_pow(power)
            im_t = draw_topomap(ax_topo, pos2d, v, cmap=cmap,
                                 vmin=0.0, vmax=max(v.max(), 1e-12), triang=triang)
        ax_topo.set_title(f'{label}\nTopo', fontsize=8, fontweight='bold', color=color)
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd = fig.add_subplot(gs[topo_row, right_col])
        ax_psd.imshow(psd_cf[::-1], aspect='auto', cmap=cmap, origin='lower',
                      vmin=0.0, vmax=max(psd_cf.max(), 1e-12),
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD', fontsize=8, fontweight='bold', color=color)
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=6)
        ax_psd.set_yticks([])

        ax_phase = fig.add_subplot(gs[topo_row + 1, right_col])
        if phase_c is None:
            ax_phase.axis('off')
        else:
            # Cyclic colormap (twilight) so a bar's color itself carries the phase,
            # not just its length/direction. invert_yaxis matches the PSD imshow
            # above it (channel 0 at TOP there too — imshow flips rows + origin=
            # 'lower', see ax_psd), so a given row means the same channel in both
            # panels — pairs 1:1, letting a phase shift across channels be read
            # directly against that channel's PSD row.
            Cc = len(phase_c)
            bar_colors = plt.get_cmap('twilight')((np.asarray(phase_c) + np.pi) / (2 * np.pi))
            ax_phase.barh(range(Cc), phase_c, color=bar_colors)
            ax_phase.axvline(0, color='gray', lw=0.6)
            ax_phase.set_xlim(-np.pi, np.pi)
            ax_phase.invert_yaxis()
            ax_phase.set_yticks([])
            ax_phase.set_xticks([-np.pi, 0, np.pi])
            ax_phase.set_xticklabels(['-π', '0', 'π'], fontsize=6)
            ax_phase.set_xlabel('Phase (rad)', fontsize=6)
            ax_phase.set_title(f'{label} Phase', fontsize=8, fontweight='bold', color=color)

    _cell(0, 0, raw_power, psd_raw, None, 'Raw', 'black')
    _cell(0, 1, recon_power, psd_recon, None, 'Full Recon', 'black')
    for block_col in range(2, n_per_row):
        topo_col, right_col = block_col * 2, block_col * 2 + 1
        fig.add_subplot(gs[0:2, topo_col]).axis('off')
        fig.add_subplot(gs[0, right_col]).axis('off')
        fig.add_subplot(gs[1, right_col]).axis('off')

    def _waveform_cell(row, block_col, sig, color, label):
        # Decoded time-domain content at ONE pinned channel (the channel with the most total
        # energy for this Q-atom, see viz.extract.extract_atom_gallery), over the whole trial.
        topo_col = block_col * 2
        ax = fig.add_subplot(gs[row, topo_col:topo_col + 2])
        if sig is None or not np.isfinite(sig).any():
            ax.axis('off')
            ax.set_title('Waveform: n/a', fontsize=7, color='gray')
            return
        ax.plot(np.arange(len(sig)), sig, color=color, linewidth=0.6)
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xlim(0, len(sig) - 1)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=5)
        ax.set_title(f'{label} Waveform (pinned channel, whole trial)',
                     fontsize=7, fontweight='bold', color=color)

    for i, q in enumerate(order):
        block_row, block_col = divmod(i, n_per_row)
        topo_row = header_rows + block_row * rows_per_block
        color = 'black'
        label = f'{unit_label} {q} ({importance[q]:.3f})'
        phase_c = phase_ch_x[:, q] if phase_ch_x is not None else None
        _cell(topo_row, block_col, psd_ch_x[:, q], psd_x[q], phase_c, label, color, signed=True)
        next_row = topo_row + 2
        if has_wave:
            _waveform_cell(next_row, block_col, waveforms[q], color, label)

    for i in range(Q, n_atom_rows * n_per_row):
        block_row, block_col = divmod(i, n_per_row)
        topo_row = header_rows + block_row * rows_per_block
        topo_col, right_col = block_col * 2, block_col * 2 + 1
        fig.add_subplot(gs[topo_row:topo_row + 2, topo_col]).axis('off')
        fig.add_subplot(gs[topo_row, right_col]).axis('off')
        fig.add_subplot(gs[topo_row + 1, right_col]).axis('off')
        next_row = topo_row + 2
        if has_wave:
            fig.add_subplot(gs[next_row, topo_col:topo_col + 2]).axis('off')

    if Q > 0:
        ax_bar = fig.add_subplot(gs[header_rows:, n_cols])
        bar_labels = [f'{unit_label[0]}{q}' for q in order]
        bars = ax_bar.barh(range(Q), importance[order], color='steelblue')
        ax_bar.set_yticks(range(Q))
        ax_bar.set_yticklabels(bar_labels, fontsize=6)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Importance', fontsize=9)
        ax_bar.set_title(f'{unit_label} Importance', fontsize=10, fontweight='bold')

    fig.text(0.5, 0.002, 'PSD y-axis: Channel (Iz -> Fp1) — log1p scale', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
