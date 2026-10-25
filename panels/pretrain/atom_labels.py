"""atom_labels: what is each Q-atom, in the language of EEG? Per atom: peak frequency and classical band, scalp topography, power spectrum
and ICLabel class probabilities (brain, muscle, eye, heart, line noise, channel noise, other) -> analysis/pretrain/atom_labels.png.

An atom is read as an ICA-style component x_c(t) = w_c s(t) on held-out dense-cap windows (>= --label-min-channels real channels, the
channels real in all of them): its topography w is the leading spatial mode of the atom's signed per-patch topography (strength
weighted, sign-free), its source s(t) is the atom's decoded content projected on w and overlap-added over the window. ICLabel's own three
features (topography, spectrum, autocorrelation) are computed from w and s with mne_icalabel's functions and fed to its network.
Approximations: the data are z-scored per trial, so the source is scaled by 10 as a stand-in for microvolts (ICLabel's spectrum feature
depends on level); windows are 2.75 s and concatenated; the topography is re-referenced to the common average. Read the classes as
a guide, not a verdict. Numbers: atom_labels.npz."""
import mne
import numpy as np
import torch

from IO.channels import get_standard_coords, resolve_canonical_channels
from model.Qtome.patches import overlap_add_patches
from panels.core.numbers import load_npz, save_npz
from panels.shared.atom_dictionary import BANDS, FS, atom_templates, band_of, peak_freq
from panels.shared.atom_readout import _signed_topo
from panels.shared.eval_windows import batches, eval_windows

PRESETS = {'standard'}
CLASSES = ('brain', 'muscle', 'eye', 'heart', 'line noise', 'channel noise', 'other')   # mne_icalabel's column order
UV = 10.0                      # shortcut: stand-in for microvolts per z-score unit (a typical EEG standard deviation)


def add_args(flags):
    flags.add('--label-windows', type=int, default=300, help='atom_labels: held-out dense-cap windows to read the atoms on')
    flags.add('--label-min-channels', type=int, default=56, help='atom_labels: a window needs this many real channels')


def select_windows(ctx):
    """The first --label-windows held-out windows with enough real channels -> (their batch indices in eval order, common channel slots)."""
    ds, idx = eval_windows(ctx.config, 4000)
    keep, common = [], None
    for ids, x, c, t, v in batches(ds, idx, bs=64):
        for k in range(len(ids)):
            if int(v[k].sum()) >= ctx.args.label_min_channels and len(keep) < ctx.args.label_windows:
                keep.append(ids[k]); common = v[k].clone() if common is None else common & v[k]
        if len(keep) >= ctx.args.label_windows:
            break
    return ds, keep, common


@torch.no_grad()
def read_atoms(model, ds, keep, slots, device, bs=32):
    """Two passes over the windows. Pass 1: per atom, the strength-weighted second moment of the signed patch topography over the
    chosen channel slots -> its leading eigenvector w (the topography). Pass 2: the atom's content projected on w, overlap-added ->
    sources [S, windows * T] and the mean strength per atom."""
    S = model.n_atoms
    slots = torch.as_tensor(slots)
    cov = torch.zeros(S, len(slots), len(slots), dtype=torch.float64)
    strength, count = torch.zeros(S, dtype=torch.float64), 0
    for ids, x, coords, t, valid in batches(ds, keep, bs):
        out = model.encode_atoms(x.to(device), coords.to(device), t.to(device), valid.to(device))
        topo = _signed_topo(out.amp.permute(0, 2, 1, 3))[..., slots.to(device)].double().cpu()      # [G, S, C*]
        h = out.h.double().cpu()                                                                    # [G, S]
        cov += torch.einsum('gs,gsc,gsd->scd', h ** 2, topo, topo)
        strength += h.sum(0); count += len(h)
    w = torch.linalg.eigh(cov)[1][..., -1]                                                          # [S, C*], unit
    w = (w * torch.sign(w[torch.arange(S), w.abs().argmax(1)])[:, None]).float()                    # sign: the largest entry positive
    sources = []
    for ids, x, coords, t, valid in batches(ds, keep, bs):
        B, C, N, L = x.shape
        out = model.encode_atoms(x.to(device), coords.to(device), t.to(device), valid.to(device))
        contrib = model.atoms.decode(out.amp)[:, slots.to(device)]                                  # [G, C*, S, L]
        src = torch.einsum('gcsl,sc->gsl', contrib, w.to(device)).reshape(B, N, S, L).permute(0, 2, 1, 3)   # [B, S, N, L]
        sources.append(overlap_add_patches(src, model.patch_stride).cpu())                          # [B, S, T]
    return w.numpy(), torch.cat(sources).permute(1, 0, 2).reshape(S, -1).numpy(), (strength / count).float().numpy()


def iclabel_probabilities(names, topo, sources):
    """ICLabel on pseudo-components: topo [C*, S] (columns = topographies), sources [S, n_times] in microvolts (200 Hz) -> psd [S, 100],
    probabilities [S, 7]. Uses mne_icalabel's feature functions directly (no ICA object: its decomposition is what the atoms replace)."""
    from types import SimpleNamespace
    from mne_icalabel.iclabel.features import _eeg_autocorr_welch, _eeg_rpsd, _eeg_topoplot
    from mne_icalabel.iclabel.network import run_iclabel
    mne.set_log_level('ERROR')
    raw = mne.io.RawArray(np.zeros((len(names), sources.shape[1])), mne.create_info(names, FS, 'eeg'), verbose=False)
    raw.set_montage('standard_1005')
    ica = SimpleNamespace(n_components_=sources.shape[0])
    icawinv = topo - topo.mean(0, keepdims=True)                                                    # common-average reference
    t = _eeg_topoplot(raw, icawinv, names) * 0.99
    psd = _eeg_rpsd(raw, ica, sources) * 0.99
    ac = _eeg_autocorr_welch(raw, ica, sources) * 0.99
    return psd[0, :, 0, :].T, run_iclabel(t, psd, ac, backend=None)


def compute(ctx):
    model, device = ctx.model, ctx.device
    names64 = resolve_canonical_channels(ctx.config['preprocess_params']['canonical_channels'])
    ds, keep, common = select_windows(ctx)
    if not keep:
        raise RuntimeError(f'no held-out window has >= {ctx.args.label_min_channels} real channels')
    slots = [i for i in torch.nonzero(common).flatten().tolist() if i < len(names64) and get_standard_coords(names64[i]) is not None]
    names = [names64[i] for i in slots]
    print(f'  {len(keep)} windows, {len(slots)} channels common to all of them')
    w, sources, strength = read_atoms(model, ds, keep, slots, device)
    psd_feature, probs = iclabel_probabilities(names, w.T, sources * UV)
    D, H, _ = atom_templates(model)
    save_npz(ctx.numbers('atom_labels.npz'), topo=w, channels=np.array(names), coords=np.stack([get_standard_coords(n) for n in names]),
             probs=probs, classes=np.array(CLASSES), peak=peak_freq(D), strength=strength, psd_feature=psd_feature,
             n_windows=np.array(len(keep)))
    best = probs.argmax(1)
    print('  ICLabel class of the atoms: ' + ', '.join(f'{c} {int((best == i).sum())}' for i, c in enumerate(CLASSES) if (best == i).any()))


def draw(ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from panels.kit.topomap import build_triangulation, draw_topomap, project_coords_2d
    z = load_npz(ctx.numbers('atom_labels.npz'))
    S = len(z['peak'])
    order = np.argsort(z['peak'], kind='stable')
    pos2d = project_coords_2d(z['coords'])
    tri = build_triangulation(pos2d)
    colors = ('#1baf7a', '#eb6834', '#2a78d6', '#9a5fd0', '#eda100', '#888888', '#cccccc')
    fig, axes = plt.subplots(S, 3, figsize=(11.5, 0.72 * S + 1.2), gridspec_kw={'width_ratios': [1.0, 1.5, 2.0]})
    freqs = np.arange(1, 101)
    for r, a in enumerate(order):
        ax_t, ax_p, ax_c = axes[r]
        v = z['topo'][a]
        draw_topomap(ax_t, pos2d, v - v.mean(), cmap='RdBu_r', vmin=-np.abs(v - v.mean()).max(), vmax=np.abs(v - v.mean()).max(), triang=tri)
        ax_t.set_ylabel(f'#{a}\n{z["peak"][a]:.0f} Hz\n{band_of(z["peak"][a])}', rotation=0, ha='right', va='center', fontsize=7)
        ax_p.plot(freqs, z['psd_feature'][a], color='k', lw=0.9)
        for _, lo, hi in BANDS[1::2]:
            ax_p.axvspan(lo, hi, color='#dddddd', alpha=0.6, lw=0)
        ax_p.set_xlim(1, 100); ax_p.set_yticks([]); ax_p.tick_params(labelsize=6)
        ax_p.axvline(z['peak'][a], color='crimson', lw=0.8, ls=':')
        ax_c.barh([0], [1], color='#f4f4f4')
        left = 0.0
        for k, p in enumerate(z['probs'][a]):
            ax_c.barh([0], [p], left=left, color=colors[k]); left += p
        k = int(z['probs'][a].argmax())
        ax_c.text(1.02, 0, f'{CLASSES[k]} {z["probs"][a][k]:.2f}', va='center', fontsize=7, transform=ax_c.get_yaxis_transform())
        ax_c.set_xlim(0, 1); ax_c.set_yticks([]); ax_c.set_xticks([])
        for ax in (ax_p, ax_c):
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        if r < S - 1:
            ax_p.set_xticklabels([])
    axes[0, 0].set_title('topography', fontsize=8); axes[0, 1].set_title('spectrum (ICLabel feature, dB)', fontsize=8)
    axes[0, 2].set_title('ICLabel class probabilities', fontsize=8)
    axes[-1, 1].set_xlabel('Hz', fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, CLASSES, ncol=7, loc='lower center', fontsize=7, frameon=False)
    fig.suptitle(f'Q-atoms as EEG components (atoms ordered by peak frequency; {int(z["n_windows"])} held-out dense-cap windows)', fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 0.93, 0.98))
    out = ctx.figure('atom_labels.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f'  -> {out}')
