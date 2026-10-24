"""One real trial as a snapshot: the stage-normalised bundle (pretrain: the full model; finetune: its frozen backbone), the numbers a
snapshot is drawn from (saved so --redraw needs no model), and the two figures: the band-filtered original-vs-reconstruction grid
and the Q-atom gallery. Shared by the snapshot (pretrain) and class_snapshots (finetune) panels."""
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from IO.event_onset import lookup_event_onset_sample
from IO.preprocessing import slice_patches
from model.Qtome.patches import overlap_add_patches
from model.Qtome.plugin import QtomeTrainer
from panels.kit.timeseries import visualize_reconstruction
from panels.kit.topomap import project_coords_2d
from panels.shared.atom_gallery import plot_atom_gallery
from panels.shared.atom_readout import extract_atom_gallery


@dataclass
class SnapshotBundle:
    """Stage-normalised input to the renderers below.
    Built by build_pretrain_bundle/build_finetune_bundle, each of which knows how to
    patchify/mask/forward its own stage; the panels know nothing about pretrain vs.
    finetune."""
    x_in: torch.Tensor           # [1, C, N, L] patches
    c_in: torch.Tensor           # [1, C, 3] coords
    t_in: torch.Tensor           # [1, N] time indices
    vc_in: torch.Tensor          # [1, C] valid-channel mask
    psd_model: nn.Module         # the model atom_by_patch/atom_gallery actually run on
                                  # (the full model for pretrain, model.backbone for finetune)
    raw_t: torch.Tensor          # [1, C, T] full-resolution raw signal
    recon_t: torch.Tensor        # [1, C, T] full-resolution reconstruction
    raw_cnl: np.ndarray          # [C, N, L] raw patches
    recon_cnl: np.ndarray        # [C, N, L] reconstruction patches
    coords: np.ndarray           # [C, 3]
    channel_names: List[str]
    valid_channels: np.ndarray   # [C] bool
    patch_len: int
    mask_np: Optional[np.ndarray] = None   # [C, N] masked-patch overlay; None: the recon is unmasked
    title_suffix: str = ''                 # e.g. ' [finetune]'
    event_onset_sec: Optional[float] = None  # real-trial event onset (s into raw_t/recon_t)
    valid_start: Optional[int] = None  # [sample idx into raw_t/recon_t's T axis]
    valid_end: Optional[int] = None    # real content lies in [valid_start, valid_end)
    subject_id: Optional[int] = None
    trial_idx: Optional[int] = None
    epoch: Optional[int] = None
    filename_tag: str = ''


def _lookup_event_onset(config, dataset, trial_idx):
    """This trial's dataset's own metadata.json event_onset_sample (see
    IO.event_onset.lookup_event_onset_sample) -> seconds, or None. Only meaningful for a
    genuine single real trial: an assembled continuous window (assemble_trials=True)
    mixes multiple real trials together with no one event to mark, so this deliberately
    returns None whenever base_dataset.assemble_trials is True rather than draw a
    misleading line. `is not None` (not truthiness) throughout -- an onset of literal 0
    is a real, legitimate value, not "not configured"."""
    base_dataset = dataset.base_dataset
    if getattr(base_dataset, 'assemble_trials', True):
        return None
    fs = config.get('preprocess_params', {}).get('sample_freq')
    if not fs:
        return None
    base_idx = trial_idx % len(base_dataset)
    ds_name = base_dataset.dataset_names[base_idx]
    onset = lookup_event_onset_sample(config, ds_name)
    return (onset / fs) if onset is not None else None


def _lookup_valid_range(dataset, trial_idx):
    """(valid_start, valid_end) sample indices into raw_t/recon_t's T axis -- real content
    lies in [valid_start, valid_end), everything outside is compile-time zero-pad."""
    base_dataset = dataset.base_dataset
    base_idx = trial_idx % len(base_dataset)
    return int(base_dataset.row_valid_start[base_idx]), int(base_dataset.row_valid_end[base_idx])


def _patchify(x, patch_len, patch_stride=None):
    """x [C, T] -> (x_patches [1, C, P, L], time_idx [1, P]). Overlapping when
    patch_stride < patch_len -- see IO/preprocessing.py's slice_patches, the same helper
    train_finetune.py's RawSource uses, so a trial patchified here matches training exactly
    instead of a naive non-overlapping P = T // patch_len reshape (which silently drops
    every patch position a stride < patch_len would have added, and gives the wrong P for
    any head whose modules need an exact patch count, e.g. LearnedTimePool)."""
    x_patches, time_idx = slice_patches(x, patch_len, patch_stride)
    return x_patches.unsqueeze(0), time_idx.unsqueeze(0)


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx, device):
    """Always runs unmasked (bool_masked_pos not passed) for a clean reconstruction
    snapshot, regardless of whether the model is currently in the Masked training stage."""
    x_patches, coords, _, time_indices, _, valid_channels = dataset[trial_idx]
    C, N, L = x_patches.shape
    pp = dataset.base_dataset.config['preprocess_params']
    fs = pp['sample_freq']
    stride = pp.get('patch_stride', L)

    x_in      = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    t_in      = time_indices.unsqueeze(0).to(device)
    vc_in     = valid_channels.unsqueeze(0).to(device)

    out = model(x_in, coords=coords_in, time_idx=t_in, valid_channels=vc_in)
    raw_stitched   = overlap_add_patches(x_patches.to(device), stride)
    recon_stitched = overlap_add_patches(out.recon[0], stride)
    T_total = raw_stitched.shape[-1]

    return {
        'raw':    raw_stitched.cpu().numpy(),
        'recon':  recon_stitched.cpu().numpy(),
        'coords': coords.numpy(),
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }


@torch.no_grad()
def build_pretrain_bundle(model, dataset, trial_idx, config, device,
                           subject_id=None, epoch=None):
    """Pretrain-stage bundle: full (masked-phase-restored) model forward, unmasked
    reconstruction, per-trial Q-atom usage for title colors. Returns (bundle, metrics)
    where metrics = {'recon_mse': ..., **QtomeTrainer().epoch_metrics(model, out)},
    matching today's BaseEpochChecker.check_pretrain's returned metrics dict exactly."""
    was_training = model.training
    model.eval()
    try:
        x_patches, coords, _mask, time_indices, _, valid_channels = dataset[trial_idx]
        x_in  = x_patches.unsqueeze(0).to(device)
        c_in  = coords.unsqueeze(0).to(device)
        t_in  = time_indices.unsqueeze(0).to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)

        data = _run_reconstruction(model, dataset, trial_idx, device)

        C, N, patch_len = x_patches.shape

        out = model(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        recon_cnl = out.recon[0].detach().cpu().numpy()

        metrics = {'recon_mse': float(np.mean((data['raw'] - data['recon']) ** 2))}
        metrics.update(QtomeTrainer().epoch_metrics(model, out))

        event_onset_sec = _lookup_event_onset(config, dataset, trial_idx)
        valid_start, valid_end = _lookup_valid_range(dataset, trial_idx)

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=model,
            raw_t=torch.from_numpy(data['raw']).unsqueeze(0),
            recon_t=torch.from_numpy(data['recon']).unsqueeze(0),
            raw_cnl=x_patches.numpy(), recon_cnl=recon_cnl,
            coords=coords.numpy(), channel_names=dataset.base_dataset.channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=None,   # recon runs unmasked: no overlay
            event_onset_sec=event_onset_sec, valid_start=valid_start, valid_end=valid_end,
            subject_id=subject_id, trial_idx=trial_idx, epoch=epoch,
        )
        return bundle, metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def build_finetune_bundle(model, dataset, trial_idx, config, device,
                           subject_id=None, tag=''):
    """Finetune-stage bundle: model.backbone forward on one patchified trial.
    tag: extra filename/title suffix (e.g. '_target2_Feet_correct') -- folded into the
    bundle's filename_tag, leading underscore, filename-safe, caller's responsibility."""
    backbone = model.backbone
    pp = config.get('preprocess_params', {})
    patch_len = pp.get('patch_length', 100)
    patch_stride = pp.get('patch_stride', patch_len)

    x_raw, coords, label, valid_channels, valid_length = dataset[trial_idx]
    x_patches, time_idx = _patchify(x_raw, patch_len, patch_stride)
    x_in = x_patches.to(device)
    c_in = coords.unsqueeze(0).to(device)
    t_in = time_idx.to(device)
    vc_in = valid_channels.unsqueeze(0).to(device)

    channel_names = dataset.base_dataset.channel_names
    C, N, L = x_patches.shape[1], x_patches.shape[2], patch_len

    was_training = model.training
    model.eval()
    try:
        out = backbone(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        raw_cnl   = x_patches[0].numpy()
        recon_cnl = out.recon[0].reshape(C, N, L).detach().cpu().numpy()

        metrics = {'recon_mse': float(np.mean((raw_cnl - recon_cnl) ** 2))}
        metrics.update(QtomeTrainer().epoch_metrics(backbone, out))

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=backbone,
            # overlap-add, not reshape(N * L): with overlapping patches (stride < L) a
            # plain reshape repeats every sample len/stride times and stretches the trace.
            raw_t=overlap_add_patches(torch.from_numpy(raw_cnl), patch_stride).unsqueeze(0),
            recon_t=overlap_add_patches(torch.from_numpy(recon_cnl), patch_stride).unsqueeze(0),
            raw_cnl=raw_cnl, recon_cnl=recon_cnl,
            coords=coords.numpy(), channel_names=channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=None,
            title_suffix=' [finetune]',
            subject_id=subject_id, trial_idx=trial_idx, filename_tag=tag,
        )
        return bundle, metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def predict_all(model, ds, patch_len, patch_stride, device):
    """A finetune model's prediction for every trial of a FinetuneDataset, one trial at a time
    -> (preds, labels) numpy arrays in dataset order."""
    preds, labels = [], []
    for i in range(len(ds)):
        x, coords, label, valid, _ = ds[i]
        xp, t = slice_patches(x, patch_len, patch_stride)
        logits = model(xp[None].to(device), coords[None].to(device), time_idx=t[None].to(device),
                       valid_channels=valid[None].to(device))[0]
        preds.append(int(logits.argmax(-1)))
        labels.append(int(label))
    return np.array(preds), np.array(labels)


# ---------- the numbers a snapshot is drawn from ----------

_META = ('channel_names', 'patch_len', 'title_suffix', 'event_onset_sec', 'valid_start', 'valid_end', 'subject_id',
         'trial_idx', 'epoch', 'filename_tag')


def snapshot_numbers(bundle, config):
    """Everything draw_recon / draw_gallery need, as {name: array} for save_npz: the bundle's signals and the Q-atom gallery readout
    (the only part that runs the model)."""
    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    viz = config.get('training_params', {}).get('visualize_params', {})
    fft_resolution = viz.get('fft_resolution', 0.2)
    (_, importance, topo, psd_x, freqs, phase_topo, waveforms) = extract_atom_gallery(
        bundle.psd_model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
        fs=fs, freq_resolution=fft_resolution)
    z = {'raw_t': bundle.raw_t.numpy(), 'recon_t': bundle.recon_t.numpy(), 'raw_cnl': bundle.raw_cnl, 'recon_cnl': bundle.recon_cnl,
         'coords': bundle.coords, 'gal_importance': importance, 'gal_topo': topo, 'gal_psd': psd_x, 'gal_freqs': freqs,
         'gal_phase': phase_topo, 'gal_waveforms': np.stack(waveforms),
         'meta': np.array(json.dumps({k: getattr(bundle, k) for k in _META})),
         'cfg': np.array(json.dumps({'preprocess_params': pp, 'training_params': {'visualize_params': viz}}))}
    if bundle.mask_np is not None:
        z['mask_np'] = bundle.mask_np
    return z


def load_snapshot(z):
    """The saved numbers (load_npz dict) -> an object with the bundle's attribute names, plus .cfg: the config parts the figures read."""
    meta = json.loads(str(z['meta']))
    return SimpleNamespace(cfg=json.loads(str(z['cfg'])), raw_t=torch.from_numpy(z['raw_t']), recon_t=torch.from_numpy(z['recon_t']),
                           raw_cnl=z['raw_cnl'], recon_cnl=z['recon_cnl'], coords=z['coords'],
                           mask_np=z.get('mask_np'), z=z, **meta)


def draw_recon(bundle, config, out_dir):
    """Band-filtered orig-vs-recon grid for one trial."""
    viz_dir = out_dir
    os.makedirs(viz_dir, exist_ok=True)

    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    bandpass = pp.get('bandpass_filter', {})
    l_freq, h_freq = bandpass.get('l_freq'), bandpass.get('h_freq')
    viz_cfg = config.get('training_params', {}).get('visualize_params', {})
    band_edges = ({name: tuple(edges) for name, edges in viz_cfg['bands'].items()}
                  if viz_cfg.get('bands') else None)

    visualize_reconstruction(
        None, (bundle.raw_t, bundle.recon_t), bundle.epoch,
        output_dir=viz_dir,
        channel_names=bundle.channel_names,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx,
        mask=bundle.mask_np, patch_len=bundle.patch_len,
        patch_stride=pp.get('patch_stride', bundle.patch_len),
        tag=bundle.filename_tag.lstrip('_') + ('_' if bundle.filename_tag else ''),
        fs=fs or 200.0, l_freq=l_freq, h_freq=h_freq, band_edges=band_edges,
        event_onset_sec=bundle.event_onset_sec,
        valid_start=bundle.valid_start, valid_end=bundle.valid_end,
    )


def draw_gallery(bundle, config, out_dir, cmap='YlOrRd'):
    """Q-atom gallery for one trial: whole-trial raw/recon PSD, then per Q-atom its topography, PSD,
    phase and waveform."""
    viz_dir = out_dir
    os.makedirs(viz_dir, exist_ok=True)

    pos2d = project_coords_2d(bundle.coords)
    epoch_tag = (f'_ep{bundle.epoch:04d}' if bundle.epoch is not None else '') + bundle.filename_tag
    tagged_epoch_tag = f'{epoch_tag}{bundle.title_suffix}'

    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    bandpass = pp.get('bandpass_filter', {})
    l_freq, h_freq = bandpass.get('l_freq'), bandpass.get('h_freq')
    viz_cfg = config.get('training_params', {}).get('visualize_params', {})
    fft_resolution = viz_cfg.get('fft_resolution', 0.2)
    psd_range = viz_cfg.get('psd_freq_range')
    psd_l_freq, psd_h_freq = tuple(psd_range) if psd_range else (l_freq, h_freq)

    # Whole-trial raw/recon PSD -- same FFT settings as the gallery's own per-atom PSD
    # (freq_resolution=fft_resolution drives both), so the header row is directly
    # comparable to the Q-atom rows below it. n_fft must match extract_atom_gallery's
    # own n_fft (round(fs/fft_resolution)); rfft's n= transparently
    # zero-pads a short trial or truncates a long one to match.
    raw_t   = bundle.raw_t[0].numpy()
    recon_t = bundle.recon_t[0].numpy()
    T = raw_t.shape[-1]
    n_fft = int(round(fs / fft_resolution)) if fs else T

    def _demean_hann_rfft_np(x):
        x = x - x.mean(axis=-1, keepdims=True)
        win = np.hanning(x.shape[-1])
        return np.fft.rfft(x * win, n=n_fft, axis=-1)

    fft_raw   = _demean_hann_rfft_np(raw_t)
    fft_recon = _demean_hann_rfft_np(recon_t)
    psd_raw   = fft_raw.real**2   + fft_raw.imag**2
    psd_recon = fft_recon.real**2 + fft_recon.imag**2

    raw_power   = (bundle.raw_cnl   ** 2).mean(axis=(1, 2))
    recon_power = (bundle.recon_cnl ** 2).mean(axis=(1, 2))

    z = bundle.z
    gal_importance, psd_ch_x_g, psd_x_g, gal_freqs, phase_ch_x_g = (z['gal_importance'], z['gal_topo'], z['gal_psd'],
                                                                    z['gal_freqs'], z['gal_phase'])
    waveforms_g = list(z['gal_waveforms'])

    if psd_l_freq is not None and psd_h_freq is not None:
        band = (gal_freqs >= psd_l_freq) & (gal_freqs <= psd_h_freq)
        gal_freqs = gal_freqs[band]
        psd_x_g = psd_x_g[:, :, band]
        psd_raw, psd_recon = psd_raw[:, band], psd_recon[:, band]

    out_path = os.path.join(
        viz_dir, f"sub{bundle.subject_id}_trial{bundle.trial_idx}{epoch_tag}_atom_gallery.png")
    plot_atom_gallery(
        out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
        psd_ch_x_g, psd_x_g, gal_freqs, gal_importance, cmap=cmap,
        phase_ch_x=phase_ch_x_g, waveforms=waveforms_g,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Atom',
    )
    print(f"  [snapshot] -> {out_path}")
