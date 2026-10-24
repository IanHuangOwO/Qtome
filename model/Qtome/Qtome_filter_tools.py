"""Task filter: a reconstruction that keeps only what classification heads use (analyze.py pretrain task_filter_* panels; history in
docs/reports/2026-10-14-task-filter.md).

A trained all-atom head reads the Q-atom gains through spatial filters. Per Q-atom, its filters span a channel subspace;
split into components that are uncorrelated on the training gains, every component cell [patch, component, Q-atom]
is scored by its class information less chance, the top share of cells is kept whole and the rest removed, and the
kept components are mapped back to the channels with their patterns. The filter is one operator per Q-atom, time-free
(averaged over the patches), [S, 64, 64] on gains of all 64 canonical channels. It is applied to a recording encoded
with its missing channels imputed (mask tokens at template coordinates: imputed_encoder) and its real channels are
read out, so one filter serves any channel layout.

Two filters, both from loso heads trained on imputed 64-channel gains (finetune cell <dataset>_impute64_loso_seed1 under
the backbone's finetune/allatom):
  single-task one dataset's heads (one task). Per fold: components from the head's two filter banks on 3000 training trials,
              cells scored, hard cut; the filter = the mean over the folds and the patches.
  multi-task  several paradigms at once (default MI BNCI2014001, P300 BNCI2014009, SSVEP Nakanishi2015). Per fold
              index i (the first n folds of every source, n = the fewest): components from every source's fold-i filters
              stacked, on the mean of the sources' covariances (N_PER trials each, so each source weighs the same
              whatever its patches per trial); each source scores every cell on its own trials, and a cell is kept if it
              is in the top share of ANY source's cells. Two corrections:
                absolute    gains scored in each trial's own scale (x the trial's standard deviation, which the model
                            divided by, over the subject's median): with per-trial normalization a target trial's larger
                            P300 made every other band look smaller, a class difference at 50, 60 and 100 Hz
                power only  Q-atoms above POWER_ONLY_HZ are scored by their log power alone: the 100 Hz Q-atom's
                            quadrature partner carries part of the slow evoked response in its signed gain
Importance (for the panel): the class information less chance of every channel cell, summed over patches, mean of the
folds [64, S] (multi: per source as a share of its total, the maximum over sources)."""
import glob
import json
import os

import numpy as np
import torch

from IO.dataset import build_dataset_from_config
from IO.loader import get_standard_coords
from IO.preprocessing import slice_patches
from model.factory import build_from_checkpoint
from model.Qtome.Qtome_modules import AtomBank, AtomExtractor, FeatureHead, overlap_add_patches

MONT = json.load(open('configs/montages.json'))['10-10']['channels']
CHANNELS = [m['label'] for m in MONT]                                     # the 64 slots, in slot order
FS, SHARES = 200, (0.5, 0.25, 0.1)                                        # sample rate; shares of cells the filter keeps
SOURCES = ('BNCI2014001', 'BNCI2014009', 'Nakanishi2015')                 # the multi-task filter's paradigms: MI, P300, SSVEP
POWER_ONLY_HZ, N_PER, N_SINGLE = 45, 1000, 3000                           # multi: power-only above; training trials per source and fold; single-task filter


def cell_dir(bb, ds):                                                     # bb = output/Qtome/<backbone>
    return f'{bb}/finetune/allatom/{ds}_impute64_loso_seed1'


def gains(bb, ds, C, N):
    """{subject: (amp [T, N, C, S, 2], labels)} from the newest feature cache folder with C channels (and N patches)."""
    for d in sorted(glob.glob(f'{bb}/pretrain/feature_cache/{ds}/*'), key=os.path.getmtime, reverse=True):
        fs = sorted(glob.glob(d + '/*.npz'))
        if np.load(fs[0])['amp'].shape[1:3] in ((N, C), (np.load(fs[0])['amp'].shape[1], C) if N is None else ()):   # N None: any patch count
            return {os.path.basename(f)[:-4]: (lambda z: (torch.from_numpy(z['amp'].astype(np.float32)), z['labels']))(np.load(f)) for f in fs}
    raise FileNotFoundError(f'no feature cache of {ds} with N {N}, C {C} under {bb}: run a finetune cell of it first')


def names(ds):
    """A dataset's channel labels in recording order and in gain order (slot order)."""
    lab = [v['label'] for v in json.load(open(f'datas/finetune/{ds}/metadata.json'))['data_metadata']['channels'].values() if isinstance(v, dict)]
    free = iter(i for i, m in enumerate(CHANNELS) if m not in lab)        # as IO/dataset.py: a channel outside
    slot = {n: CHANNELS.index(n) if n in CHANNELS else next(free) for n in lab}   # the montage takes the next free slot
    return lab, sorted(lab, key=slot.get)


def components(W, amp, shrink=0.05, eps=1e-4):
    """W [M, C] stacked filters, training gains [B, N, C, S, 2] (or a list of such, one per dataset) -> per Q-atom:
    unmix [S, M, C] (gains -> components that are uncorrelated with unit variance on the training gains), mix [S, M, M]
    (components -> filter outputs), pattern [S, C, M] (components -> channels); M is cut to the number of channels if
    there are fewer channels than filters. The filters themselves are far from orthogonal, and dropping one of two
    nearly parallel sources leaves a huge uncancelled waveform; uncorrelated components do not have that."""
    covs = []                                                             # a list of datasets: the mean of their covariances, so each
    for a in (amp if isinstance(amp, list) else [amp]):                   # weighs the same whatever its trials and patches per trial
        X = a.permute(3, 0, 1, 4, 2).reshape(a.shape[3], -1, a.shape[2]).double()   # [S, rows, C]
        covs.append(X.transpose(1, 2) @ X / X.shape[1])
    cov = torch.stack(covs).mean(0)
    cov = (1 - shrink) * cov + shrink * torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
    Wd = W.double(); lam, U = torch.linalg.eigh(Wd @ cov @ Wd.T)         # [S, M], [S, M, M], ascending
    r = min(W.shape)                                                      # more filters than channels: only C components exist;
    lam, U = lam[:, -r:], U[:, :, -r:]                                    # the empty ones would fill the mask's share with zeros
    lam = lam.clamp_min(0) + eps * lam.mean(-1, keepdim=True)
    unmix = (U / lam.sqrt()[:, None]).transpose(1, 2) @ Wd
    mix = U * lam.sqrt()[:, None]
    return unmix.float(), mix.float(), (cov @ unmix.transpose(1, 2)).float()


def r2(X, y):
    """X [trials, ...], labels y -> share of variance explained by the class, per remaining axis (any number of classes)."""
    mu = X.mean(0); num = sum((y == k).sum() * (X[y == k].mean(0) - mu) ** 2 for k in np.unique(y))
    return num / (((X - mu) ** 2).sum(0) + 1e-20)


def class_info(u, y, chance=True, power_only=None):
    """[N, M, S] class information of every cell of u [B, N, M, S, 2]: R2 of a + of b + of log(a^2 + b^2); chance: each
    less its chance level (k - 1) / (n - 1) for k classes and n trials, floored at 0. power_only [S] bool: those Q-atoms
    are scored by their log power alone."""
    f = torch.cat([u, torch.log(u.pow(2).sum(-1, keepdim=True) + 1e-12)], -1).double().numpy(); r = r2(f, y.numpy())
    if power_only is not None: r[..., np.asarray(power_only), :2] = 0          # [N, M, S, 3]: drop a and b of those Q-atoms
    if chance: r = np.maximum(r - (len(np.unique(y.numpy())) - 1) / (len(y) - 1), 0)   # expected R2 of a feature unrelated to the class
    return torch.from_numpy(r.sum(-1)).float()


def hard(R, q):                                                           # 1 for the top share q of cells, 0 for the rest
    return (R >= torch.quantile(R.flatten(), 1 - q).clamp_min(1e-12)).float()   # a score with many exact zeros would keep them all


def load_head(cell, fold):
    hk = torch.load(f'{cell}/finetune/run_{fold}/head.pth', map_location='cpu', weights_only=False)
    hc = hk['head_config']; head = FeatureHead(hc); head.load_state_dict(hk['model_state_dict']); head.eval()
    assert hc['num_channels'] == 64, f'{cell}: head on {hc["num_channels"]} channels, not the imputed 64'
    return hc, torch.cat([head.spatials[b].weight for b in head.entries]).detach()   # stacked filters [2K, 64]


def single_task_filter(bb, ds, rng, shares=SHARES):
    """{share: [S, 64, 64]}, importance [64, S]."""
    cell = cell_dir(bb, ds); data, ops, imp = None, {q: [] for q in shares}, []
    for fold, info in json.load(open(f'{cell}/artifacts/group_eval.json')).items():
        hc, W = load_head(cell, fold)
        data = data or gains(bb, ds, hc['num_channels'], hc['num_patches'])
        tr = [data[s] for s in info['train_subjects']]                    # newest cache of that shape: it must hold this head's classes
        assert set(np.concatenate([t[1] for t in tr]).tolist()) == set(range(hc['num_classes'])), 'the gain cache does not hold the classes of this head'
        k = rng.permutation(sum(len(t[1]) for t in tr))[:N_SINGLE]
        atr, ytr = torch.cat([t[0] for t in tr])[k], torch.from_numpy(np.concatenate([t[1] for t in tr])[k])
        imp.append(class_info(atr, ytr).sum(0))
        unmix, mix, A = components(W, atr)
        R = class_info(torch.einsum('smc,bncsx->bnmsx', unmix, atr), ytr)
        for q in shares: ops[q].append(torch.einsum('scm,nms,smd->nscd', A, hard(R, q), unmix))
    return {q: torch.stack(v).mean(0).mean(0) for q, v in ops.items()}, torch.stack(imp).mean(0)   # folds, then patches


def absolute(ds, data):
    """{subject: (gains in each trial's own scale relative to the subject's median trial, labels)}."""
    out = {}
    for s, (amp, y) in data.items():
        z = np.load(sorted(glob.glob(f'datas/finetune/{ds}/cache/{s}_fs200*.npz'))[0]); assert np.array_equal(z['labels'], y)
        sd = z['data'].astype(np.float64).std((1, 2)); out[s] = (amp * torch.from_numpy(sd / np.median(sd)).float()[:, None, None, None, None], y)
    return out


def multi_filter(bb, sources, freqs, rng, shares=SHARES, log=print):
    """{share: [S, 64, 64]}, importance [64, S]; freqs [S] the Q-atoms' dominant frequencies."""
    cells = {s: cell_dir(bb, s) for s in sources}
    ge = {s: list(json.load(open(f'{c}/artifacts/group_eval.json')).items()) for s, c in cells.items()}
    n = min(len(v) for v in ge.values()); data, ops, imp = {}, {q: [] for q in shares}, []; po = torch.from_numpy(freqs > POWER_ONLY_HZ)
    for i in range(n):
        Ws, tr = [], {}
        for s in sources:
            fold, info = ge[s][i]; hc, W = load_head(cells[s], fold)
            if s not in data: data[s] = absolute(s, gains(bb, s, 64, hc['num_patches']))
            t = [data[s][x] for x in info['train_subjects']]; k = rng.permutation(sum(len(a[1]) for a in t))[:N_PER]
            tr[s] = (torch.cat([a[0] for a in t])[k], torch.from_numpy(np.concatenate([a[1] for a in t])[k]))
            assert set(tr[s][1].tolist()) == set(range(hc['num_classes'])), f'{s}: the gain cache does not hold the classes of this head'
            Ws.append(W)
        unmix, mix, A = components(torch.cat(Ws), [a for a, _ in tr.values()])
        keep = {q: torch.zeros(unmix.shape[1], unmix.shape[0], dtype=torch.bool) for q in shares}   # [M, S]
        for s, (a, y) in tr.items():
            R = class_info(torch.einsum('smc,bncsx->bnmsx', unmix, a), y, power_only=po).mean(0)   # [M, S]
            for q in shares: keep[q] |= hard(R, q).bool()
        for q in shares: ops[q].append(torch.einsum('scm,ms,smd->scd', A, keep[q].float(), unmix))
        per = [class_info(a, y, power_only=po).sum(0) for a, y in tr.values()]   # [64, S] per source
        imp.append(torch.stack([p / p.sum().clamp_min(1e-12) for p in per]).max(0).values)
        log(f'  multi-task filter, fold index {i}: {unmix.shape[1]} components; kept share of cells ' +
            ', '.join(f'{q:g}: {keep[q].float().mean():.2f}' for q in shares))
    return {q: torch.stack(v).mean(0) for q, v in ops.items()}, torch.stack(imp).mean(0)


class Backbone:
    """The frozen backbone of output/Qtome/<backbone>: the model, its Q-atom templates D and quadrature partners H,
    their dominant frequencies, and decoding of gains [B, N, C, S, 2] to waveforms [B, C, T]."""

    def __init__(self, bb):
        ck = torch.load(f'{bb}/pretrain/checkpoint/last.pth', map_location='cpu', weights_only=False)
        self.model = build_from_checkpoint(ck).eval()
        D = ck['model_state_dict']['atoms.D']; P = np.abs(np.fft.rfft((D - D.mean(1, keepdim=True)).numpy(), axis=1)) ** 2
        self.freqs = np.fft.rfftfreq(D.shape[1], 1 / FS)[P.argmax(1)]    # [S] peak of each template's spectrum
        D = torch.nn.functional.normalize(D, dim=-1); self.D, self.H = D.numpy(), AtomBank._quadrature(D).numpy()

    def patches(self, g):                                                 # gains [B, N, C, S, 2] -> patches [B, C, N, L]
        g = g.numpy() if torch.is_tensor(g) else g
        return np.einsum('tncs,sl->tcnl', g[..., 0], self.D) + np.einsum('tncs,sl->tcnl', g[..., 1], self.H)

    def decode(self, g):                                                  # -> [B, C, T], overlap-add at the 50% patch stride
        return overlap_add_patches(torch.from_numpy(self.patches(g)), self.D.shape[1] // 2).numpy()


def layout(bb, ds):
    """Valid slots, coordinates and preprocessing of a dataset, from the config of its loso cell (or, for a dataset
    without one, a P300 cell's with only the dataset swapped); missing slots at their template position if the cell
    imputes them."""
    path = f'{bb}/finetune/allatom/{ds}_loso_seed1/artifacts/config.json'
    if os.path.exists(path):
        cfg = json.load(open(path))
    else:
        cfg = json.load(open(f'{bb}/finetune/allatom/BNCI2014009_loso_seed1/artifacts/config.json'))
        cfg['dataset_params']['finetune'] = {ds: {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['1'], 'channels_to_use': ['all']}}
    return _layout(cfg, ds)


def layout64(bb, ds):                                                     # the 64 imputed slots of an imputed cell
    return _layout(json.load(open(f'{cell_dir(bb, ds)}/artifacts/config.json')), ds)


def _layout(cfg, ds):
    d = cfg['dataset_params']['finetune'][ds]
    d['subject_to_use'] = [d['subject_to_use'][0] if d['subject_to_use'] != ['all'] else '1']
    base = build_dataset_from_config(cfg, mode='finetune').base_dataset; valid, coords = base.all_valid_channels[0].clone(), base.all_coords[0].clone()
    if d.get('impute_missing_channels'):                                  # as cache_feature.py: missing slots at their standard position
        for i in torch.nonzero(~valid).flatten().tolist(): coords[i] = torch.as_tensor(get_standard_coords(base.channel_names[i]), dtype=coords.dtype)
        valid[:] = True
    return valid, coords, cfg['preprocess_params']


def imputed_encoder(bb, ds, model, coords64):
    """encode(w [B, C, T] of ds's real channels in gain order, impute) -> gains in trial units [B, N, C or 64, S, 2]: of
    the real channels only, or (impute) of all 64 slots, the missing ones as mask tokens at their template coordinates."""
    valid, coords, pp = layout(bb, ds); idx = torch.nonzero(valid).flatten()
    assert [CHANNELS[i] for i in idx.tolist()] == names(ds)[1] and torch.allclose(coords[idx], coords64[idx], atol=1e-5)
    ext, ext64 = AtomExtractor(model, idx.tolist()).eval(), AtomExtractor(model, list(range(64))).eval()

    def encode(w, impute):
        w = torch.from_numpy(w).float(); mu, s_ = w.mean((1, 2), keepdim=True), w.std((1, 2), keepdim=True)
        x = torch.zeros(len(w), len(valid), w.shape[-1]); x[:, idx] = (w - mu) / s_; out = []
        with torch.no_grad():
            for i in range(0, len(w), 64):
                xp, _ = slice_patches(x[i:i + 64], pp['patch_length'], pp['patch_stride']); b, P = xp.shape[0], xp.shape[2]
                out.append((ext64 if impute else ext)(xp, coords64[None].expand(b, -1, -1), torch.arange(P)[None].expand(b, P), valid[None].expand(b, -1), impute_missing=impute))
        return torch.cat(out) * s_[:, :, :, None, None]
    return encode


def build(bb, source, backbone, sources=SOURCES, seed=0, log=print):
    """The filter of source ('multi' or a dataset name) as a dict saved by the task_filter panels: filters {share: [S, 64, 64]},
    importance [64, S], coords64 [64, 3]."""
    rng = np.random.default_rng(seed)
    if source == 'multi':
        filters, imp = multi_filter(bb, list(sources), backbone.freqs, rng, log=log)
    else:
        filters, imp = single_task_filter(bb, source, rng)
    _, coords64, _ = layout64(bb, sources[0] if source == 'multi' else source)
    return {'source': source, 'sources': list(sources) if source == 'multi' else [source], 'shares': list(SHARES),
            'filters': filters, 'importance': imp, 'coords64': coords64}


def _selfcheck():
    """Components are uncorrelated with unit variance, mix gives back the filter outputs, keeping every component
    leaves the filter outputs unchanged; class information finds the one cell that depends on the class."""
    g = torch.Generator().manual_seed(0); W, amp = torch.randn(6, 10, generator=g), torch.randn(400, 4, 10, 3, 2, generator=g)
    unmix, mix, A = components(W, amp, shrink=0.0, eps=1e-9)
    u = torch.einsum('smc,bncsx->bnmsx', unmix, amp)
    assert torch.allclose(torch.einsum('bnmsx,bnksx->smk', u, u) / (400 * 4 * 2), torch.eye(6).expand(3, -1, -1), atol=1e-2)
    assert torch.allclose(torch.einsum('skm,bnmsx->bnksx', mix, u), torch.einsum('kc,bncsx->bnksx', W, amp), atol=1e-3)
    assert torch.allclose(W @ A @ unmix, W.expand(3, -1, -1), atol=1e-2)
    y = torch.arange(4000) % 2; x = torch.randn(4000, 1, 2, 1, 2, generator=g); x[:, :, 0, :, 0] += 3 * y[:, None, None]
    ci = class_info(x, y); assert ci[0, 0, 0] > 0.5 and ci[0, 1, 0] < 0.01
    assert hard(torch.tensor([0.0, 1, 2, 3]), 0.5).tolist() == [0, 0, 1, 1]
