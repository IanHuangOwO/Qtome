"""
Backbone evaluation on held-out pretrain windows. The model runs on --device (GPU by default); the baselines, spectra and bookkeeping
stay on the CPU. Every backbone sees the SAME
windows and the SAME masks (seeded per window), so runs trained with different masking
schemes are compared on one task.

1. Test masks -- masked-patch MSE (plain patch MSE on masked real tokens, as _recon_loss's
   l_masked) for each mask type, next to two model-free baselines:
     idw     inverse-distance interpolation from the 4 nearest visible channels (channel masks)
     linear  linear interpolation over time across the hidden samples (time masks)
     ar      least-squares AR(32) interpolation per channel from both sides of the hole (time blocks)
     zero    predicting 0 (= signal power of the normalised data)
   Types: token_runs (random 3-patch runs, 50%: the baseline's own task), random_channel
   (30% of channels), channel_cluster (30%), time_block (30% of time), time_block_short (one hole of
   4 patches, 0.75 s hidden at stride 50), time_block_3 / time_block_2 (one hole of 3 / 2 patches), and motor3_to_bci22
   (dense caps only: every channel except C3/Cz/C4 hidden, scored on the other 19 BCI-22
   channels -- the sparse-cap imputation BNCI2014004 needs). Also split sparse (<= 22 real
   channels) vs dense windows. For the time masks also 'hidden': scored only on the samples no visible
   patch covers (the per-patch score includes the edge halves a visible neighbour shows).
2. Ablations under every test mask: coords shuffled across channels, all channels at the mean
   position, time position removed (time_off: every patch at position 0) or shuffled over the patches
   (time_shuffle), and coordinate jitter (2 / 5 / 10 mm per channel) and left-right mirror (feature_cache.
   transform_coords). Masked MSE per mask type;
   ablation_masked_mse keeps the token_runs row for older readers.
3. Structure: coordinate-embedding similarity vs electrode closeness (Spearman, 10-10
   channels); with a RelativeSpatialBias, per block
   the Spearman correlation of the head-averaged bias with closeness (> 0: prefers
   neighbours) and its mean |value|.
4. Masked spectrum, per test mask: log-spectral distance on the masked samples (multi-resolution
   log-magnitude STFT, the measure of the rejected STFT loss); per band the relative
   error |X^ - X|^2 / |X|^2 (phase-sensitive: 0 = exact, 1 = as bad as predicting 0), also for the
   unmasked reconstruction; and per band predicted / true power (1 = right power;
   << 1 = the prediction shrinks toward 0, what a time-domain MSE target does to an
   unpredictable-phase rhythm).
5. Seam disagreement (unmasked): the squared difference between two neighbouring patches'
   reconstructions on the samples they share (50% overlap), / signal power there. 0 = the patches
   agree wherever they overlap; what the trial-level MSE term enforces.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from IO.channels import load_montage_channels
from IO.channels import get_standard_coords
from model.Qtome.finetune.feature_cache import transform_coords
from panels.shared.eval_windows import batches, eval_windows, make_mask
from model.Qtome.embeddings import fourier_features
from model.Qtome.patches import overlap_add_patches


SPEC_BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45),
              'high': (45, 90)}                                   # 'high' added 2026-10-17 (whitened-MSE card)


LOG_FLOOR = 0.1   # added to |X| before the log, so near-silent bins don't dominate


def log_spectral_distance(rec_t, x_t, w_t, sizes=(32, 64, 128)):
    """rec_t / x_t / w_t [B, C, T] -> mean over sizes N of the frame-weighted mean
    |log(|X^|+f) - log(|X|+f)| (Hann window N, hop N/4, no centering; frame weight = mean sample
    weight). 0 if no frame counts."""
    B, C, T = x_t.shape
    rec, tgt, wt = (t.reshape(B * C, T).float() for t in (rec_t, x_t, w_t))
    out = []
    for n in sizes:
        hop, win = max(n // 4, 1), torch.hann_window(n)
        mag = lambda s: torch.stft(s, n_fft=n, hop_length=hop, window=win, center=False, return_complex=True).abs()
        d = ((mag(rec) + LOG_FLOOR).log() - (mag(tgt) + LOG_FLOOR).log()).abs().mean(1)   # [BC, M]
        fw = wt.unfold(-1, n, hop).mean(-1)
        out.append(float((d * fw).sum() / fw.sum()) if fw.sum() > 0 else 0.0)
    return sum(out) / len(out)


def masked_spectrum(recon, x, score, stride, fs, n_fft=128):
    """recon / x [B, C, N, L], score [B, C, N] bool (scored tokens) -> (lsd * weight, weight,
    {band: predicted power}, {band: true power}, {band: error power}) summed over the batch, on the
    overlap-added trial with frames weighted by their share of scored samples."""
    rec_t, x_t = overlap_add_patches(recon.float(), stride), overlap_add_patches(x, stride)
    w_t = overlap_add_patches(score[..., None].expand_as(x).float(), stride)          # [B, C, T]
    weight = float(w_t.sum())
    lsd = log_spectral_distance(rec_t, x_t, w_t) * weight
    B, C, T = x_t.shape
    win, hop = torch.hann_window(n_fft), n_fft // 4
    stft = lambda s: torch.stft(s.reshape(B * C, T), n_fft=n_fft, hop_length=hop, window=win,
                                center=False, return_complex=True)                  # [BC, F, M]
    fw = w_t.reshape(B * C, T).unfold(-1, n_fft, hop).mean(-1)[:, None, :]           # [BC, 1, M]
    Xr, Xx = stft(rec_t), stft(x_t)
    Pr, Px, Pe = (P.abs().pow(2) * fw for P in (Xr, Xx, Xr - Xx))
    f = torch.fft.rfftfreq(n_fft, 1 / fs)
    band = lambda P, lo, hi: float(P[:, (f >= lo) & (f < hi)].sum())
    return (lsd, weight, *({k: band(P, *r) for k, r in SPEC_BANDS.items()} for P in (Pr, Px, Pe)))


# ---------- test masks ----------

def idw_predict(x, coords, visible_ch, k=4):
    """x [C, N, L]; predict every channel from the k nearest visible channels (1/d^2)."""
    vis = visible_ch.nonzero().flatten()
    if len(vis) == 0:
        return torch.zeros_like(x)
    d = torch.cdist(coords, coords[vis]).clamp_min(1e-3)                 # [C, V]
    k = min(k, len(vis))
    dk, ik = d.topk(k, dim=1, largest=False)
    w = 1.0 / dk.pow(2); w = w / w.sum(1, keepdim=True)                   # [C, k]
    return torch.einsum('ck,cknl->cnl', w, x[vis][ik])


def linear_time_predict(x, m, stride):
    """Overlap-add the visible patches to a signal, linearly interpolate the samples no visible
    patch covers, slice back into patches. x, m: [C, N, L] / [C, N]."""
    C, N, L = x.shape
    sig = overlap_add_patches(x[None], stride)[0]                          # [C, T]
    cover = overlap_add_patches((~m)[None, :, :, None].float().expand(1, C, N, L), stride)[0] > 1e-6
    out = sig.clone()
    t = np.arange(sig.shape[1])
    for c in range(C):
        hid = ~cover[c]
        if hid.any() and (~hid).any():
            out[c, hid] = torch.from_numpy(np.interp(t[hid.numpy()], t[~hid.numpy()], sig[c, ~hid].numpy())).float()
    idx = torch.arange(N)[:, None] * stride + torch.arange(L)[None, :]
    return out[:, idx]                                                    # [C, N, L]


def ar_time_predict(x, m, stride, order=32):
    """The linear reference for a time hole: per channel, fit an AR(order) model on the visible samples
    of this window and fill the hidden ones with the values that minimise its prediction residual
    given both sides (least-squares AR interpolation, Janssen et al. 1986). Unlike the straight line it
    continues rhythms. Falls back to the straight line where too few samples are visible to fit.
    x, m: [C, N, L] / [C, N]."""
    C, N, L = x.shape
    sig = overlap_add_patches(x[None], stride)[0].double().numpy()         # [C, T]
    cover = (overlap_add_patches((~m)[None, :, :, None].float().expand(1, C, N, L), stride)[0] > 1e-6).numpy()
    out = overlap_add_patches(linear_time_predict(x, m, stride)[None], stride)[0].double().numpy()
    T, p = sig.shape[1], order
    lag = np.arange(p + 1)
    for c in range(C):
        hid = ~cover[c]
        if not hid.any() or not sig[c].any():
            continue
        rows = np.arange(p, T)
        touch = hid[rows[:, None] - lag[None, :]]                          # [T - p, p + 1]: sample t - k hidden
        fit = rows[~touch.any(1)]
        var = sig[c][cover[c]].var()
        if len(fit) < 4 * p or var < 1e-10:                                 # too little to fit, or a flat channel
            continue
        X = sig[c][fit[:, None] - lag[None, 1:]]
        coef = np.linalg.lstsq(X.T @ X + 1e-3 * len(fit) * var * np.eye(p), X.T @ sig[c][fit], rcond=None)[0]
        use = rows[touch.any(1)]
        A = np.zeros((len(use), T))
        A[np.arange(len(use))[:, None], use[:, None] - lag[None, :]] = np.concatenate([[1.0], -coef])[None, :]
        # Residuals alone leave a hole at the window edge to an unstable backward recursion; the weak prior
        # x ~ N(0, var) (rows scaled by the innovation std) bounds it and shrinks unpredictable samples to 0.
        Ah = A[:, hid]
        innov = max(float(np.mean((sig[c][fit] - X @ coef) ** 2)), 1e-6 * var)
        out[c, hid] = np.linalg.solve(Ah.T @ Ah + innov / var * np.eye(Ah.shape[1]), -Ah.T @ (A[:, ~hid] @ sig[c, ~hid]))
    idx = torch.arange(N)[:, None] * stride + torch.arange(L)[None, :]
    return torch.from_numpy(out).float()[:, idx]                           # [C, N, L]


KINDS = ['token_runs', 'random_channel', 'channel_cluster', 'time_block', 'motor3_to_bci22', 'time_block_short',
         'time_block_3', 'time_block_2']


# ---------- ablations ----------

def ablate(name, coords, t, valid):
    coords, t = coords.clone(), t.clone()
    for b in range(coords.shape[0]):
        v = valid[b].nonzero().flatten()
        if name == 'coords_shuffle':
            coords[b, v] = coords[b, v[torch.randperm(len(v))]]
        elif name == 'coords_mean':
            coords[b, v] = coords[b, v].mean(0)
    if name.startswith('coords_jitter') or name == 'coords_mirror':
        coords = torch.stack([transform_coords(c, name[len('coords_'):], seed=b) for b, c in enumerate(coords)])
    return coords, t


class time_position:
    """Context manager for the time-position ablations: while inside, FoPE gets every patch at position 0
    ('time_off': temporal attention no longer knows order or distance) or the positions in one fixed random order
    ('time_shuffle'). Any other name changes nothing."""
    def __init__(self, model, name):
        self.fope, self.name = model.encoder.fope, name

    def __enter__(self):
        fwd = self.fope.forward
        if self.name == 'time_off':
            self.fope.forward = lambda pos: fwd(torch.zeros_like(pos))
        elif self.name == 'time_shuffle':
            self.fope.forward = lambda pos: fwd(pos[torch.randperm(len(pos), generator=torch.Generator().manual_seed(0)).to(pos.device)])

    def __exit__(self, *exc):
        self.fope.__dict__.pop('forward', None)


ABLATIONS = ['baseline', 'coords_shuffle', 'coords_mean', 'time_off', 'time_shuffle',   # time through FoPE (time_idx is unused)
             'coords_jitter_2mm', 'coords_jitter_5mm', 'coords_jitter_10mm', 'coords_mirror']


def run_model(model, device, x, coords, t, **kw):
    """One forward on `device`; the output's recon comes back on the CPU, where the scoring runs."""
    kw = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kw.items()}
    out = model(x.to(device), coords.to(device), None if t is None else t.to(device), **kw)
    out.recon = out.recon.cpu()
    return out


@torch.no_grad()
def evaluate(model, config, out_path, max_windows=512, name='', device='cpu'):
    """Runs every test mask, ablation and structure check on the backbone's held-out windows, writes
    the summary JSON to out_path (compare-able across backbones: same windows, same masks), prints it."""
    model = model.to(device).eval()
    stride = config['preprocess_params'].get('patch_stride', config['preprocess_params']['patch_length'])
    ds, idx = eval_windows(config, max_windows)
    bd = ds.base_dataset
    norm = bd._normalize_label
    n2i = {norm(n): i for i, n in enumerate(bd.channel_names)}
    names_idx = {m: [n2i[norm(n)] for n in load_montage_channels(m) if norm(n) in n2i] for m in ('motor-3', 'bci-22')}

    acc = {}                                              # (kind, group, predictor) -> [sum, count]
    def add(key, err, sel):
        a = acc.setdefault(key, [0.0, 0])
        a[0] += float(err[sel].sum()); a[1] += int(sel.sum())

    abl = {k: {a: [0.0, 0] for a in ABLATIONS} for k in KINDS}

    def add_spectrum(sp, stats):
        lsd, wsum, pr, px, pe = stats
        sp[0] += lsd; sp[1] += wsum
        for bnd in SPEC_BANDS:
            sp[2][bnd] += pr[bnd]; sp[3][bnd] += px[bnd]; sp[4][bnd] += pe[bnd]
    spec = {k: [0.0, 0.0] + [dict.fromkeys(SPEC_BANDS, 0.0) for _ in range(3)] for k in KINDS + ['unmasked'] + [k + '|ar' for k in KINDS if k.startswith('time_block')]}
    fs = float(config['preprocess_params']['sample_freq'])
    seam = [0.0, 0.0]                                     # disagreement sum, signal power sum
    for wids, x, coords, t, valid in batches(ds, idx):
        B, C, N, L = x.shape
        r = run_model(model, device, x, coords, t, valid_channels=valid).recon.float()         # unmasked
        add_spectrum(spec['unmasked'], masked_spectrum(r, x, (x.abs().amax(-1) > 0) & valid[:, :, None], stride, fs))
        ov = L - stride
        real = (x[:, :, :-1].abs().amax(-1) > 0) & (x[:, :, 1:].abs().amax(-1) > 0) & valid[:, :, None]
        seam[0] += float(((r[:, :, :-1, stride:] - r[:, :, 1:, :ov]).pow(2).mean(-1) * real).sum())
        seam[1] += float((x[:, :, 1:, :ov].pow(2).mean(-1) * real).sum())
        valid_tok = torch.stack([ds._valid_masks[w].view(C, N) for w in wids])
        for kind in KINDS:
            masks, scores = [], []
            for b in range(B):
                m, s = make_mask(kind, valid_tok[b], coords[b], wids[b], names_idx,
                                 bd.all_named_slots[bd.trial_to_coords_idx[wids[b]]])
                masks.append(torch.zeros(C, N, dtype=torch.bool) if m is None else m)
                scores.append(torch.zeros(C, N, dtype=torch.bool) if s is None else s)
            mp, sc = torch.stack(masks), torch.stack(scores)
            if not sc.any():
                continue
            out = run_model(model, device, x, coords, t, bool_masked_pos=mp, valid_channels=valid)
            se_model = (out.recon.float() - x).pow(2)                    # [B, C, N, L]
            add_spectrum(spec[kind], masked_spectrum(out.recon, x, sc, stride, fs))
            ar_rec = x.clone()                                           # AR reference, time holes only
            for b in range(B):
                if not sc[b].any():
                    continue
                group = 'sparse' if int(valid[b].sum()) <= 22 else 'dense'
                preds = {'model': se_model[b], 'zero': x[b].pow(2)}          # squared error per sample [C, N, L]
                if kind in ('random_channel', 'channel_cluster', 'motor3_to_bci22'):
                    vis_ch = valid[b] & ~mp[b].any(1)
                    preds['idw'] = (idw_predict(x[b], coords[b], vis_ch) - x[b]).pow(2)
                if kind.startswith('time_block') or kind == 'token_runs':
                    preds['linear'] = (linear_time_predict(x[b], mp[b], stride) - x[b]).pow(2)
                if kind.startswith('time_block'):
                    ar_rec[b] = ar_time_predict(x[b], mp[b], stride)
                    preds['ar'] = (ar_rec[b] - x[b]).pow(2)
                # 'hidden': only the samples no visible patch covers. A masked patch at a hole's edge shares
                # half its samples with a visible neighbour (50% overlap), so the per-patch score is diluted.
                hid = None
                if kind.startswith('time_block') or kind == 'token_runs':
                    cover = overlap_add_patches((~mp[b])[None, :, :, None].float().expand(1, C, N, L), stride)[0] > 1e-6
                    hid = (~cover)[:, torch.arange(N)[:, None] * stride + torch.arange(L)[None, :]] & sc[b][:, :, None]
                for p, se in preds.items():
                    e = se.mean(-1)
                    add((kind, 'all', p), e, sc[b]); add((kind, group, p), e, sc[b])
                    if hid is not None:
                        add((kind, 'hidden', p), se, hid)
            if kind.startswith('time_block'):
                add_spectrum(spec[kind + '|ar'], masked_spectrum(ar_rec, x, sc, stride, fs))
            for a in ABLATIONS:
                cc, tt = ablate(a, coords, t, valid)
                torch.manual_seed(0)
                with time_position(model, a):
                    o = run_model(model, device, x, cc, tt, bool_masked_pos=mp, valid_channels=valid)
                e = (o.recon.float() - x).pow(2).mean(-1)
                abl[kind][a][0] += float(e[sc].sum()); abl[kind][a][1] += int(sc.sum())

    # structure (CPU: the positions are CPU tensors)
    model = model.cpu()
    emb = model.embed
    labels = [n for n in load_montage_channels('10-10') if get_standard_coords(n) is not None]
    pos = torch.tensor(np.stack([get_standard_coords(n) for n in labels]), dtype=torch.float32)
    iu = torch.triu_indices(len(labels), len(labels), 1)
    closeness = -torch.cdist(pos, pos)
    struct = {'spatial_embedding': emb.coord_proj is not None}
    if emb.coord_proj is not None:
        ce = emb.coord_proj(fourier_features(pos))
        ce = ce / ce.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        struct['coord_sim_vs_closeness_spearman'] = float(spearmanr((ce @ ce.T)[iu[0], iu[1]], closeness[iu[0], iu[1]]).correlation)
    if getattr(model, 'spatial_bias', None) is not None:
        bias = model.spatial_bias(pos[None])[0].mean(1)                    # [depth, C, C], heads averaged
        off = ~torch.eye(len(labels), dtype=torch.bool)
        struct['spatial_bias_per_block'] = [
            {'closeness_spearman': float(spearmanr(bias[i][off], closeness[off]).correlation),
             'mean_abs': float(bias[i][off].abs().mean())} for i in range(bias.shape[0])]

    res = {'windows': len(idx),
           'test_masks': {f'{k}|{g}|{p}': v[0] / max(v[1], 1) for (k, g, p), v in acc.items()},
           'ablation_masked_mse': {a: v[0] / max(v[1], 1) for a, v in abl['token_runs'].items()},
           'ablation_by_mask': {k: {a: v[0] / max(v[1], 1) for a, v in d.items()} for k, d in abl.items()},
           'masked_spectrum': {k: {'lsd': sp[0] / sp[1],
                                   'power_ratio': {b: sp[2][b] / max(sp[3][b], 1e-12) for b in SPEC_BANDS},
                                   'error_ratio': {b: sp[4][b] / max(sp[3][b], 1e-12) for b in SPEC_BANDS}}
                               for k, sp in spec.items() if sp[1] > 0},
           'seam_disagreement': seam[0] / max(seam[1], 1e-12),
           'structure': struct}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, 'w'), indent=2)

    print(f'\n=== {name} ({len(idx)} held-out windows)')
    print(f'  {"test mask":17} {"group":6} {"model":>7} {"idw":>7} {"linear":>7} {"zero":>7}')
    for k in KINDS:
        for g in ('all', 'sparse', 'dense'):
            row = {p: res['test_masks'].get(f'{k}|{g}|{p}') for p in ('model', 'idw', 'linear', 'ar', 'zero')}
            if row['model'] is None:
                continue
            print(f'  {k:17} {g:6} ' + ' '.join(f'{v:7.4f}' if v is not None else f'{"-":>7}' for v in row.values()))
    print(f'  ablation, masked MSE change: {"mask":17} ' + ' '.join(f'{a:>15}' for a in ABLATIONS[1:]))
    for k, d in res['ablation_by_mask'].items():
        if d['baseline'] > 0:                           # 0 when no held-out window scored this mask kind
            print(f'  {"":28} {k:17} ' + ' '.join(f'{d[a] / d["baseline"] - 1:>+15.0%}' for a in ABLATIONS[1:]))
    print(f'  masked spectrum: {"mask":17} {"lsd":>6}  predicted/true power ' + ' '.join(f'{b:>6}' for b in SPEC_BANDS))
    for k, d in res['masked_spectrum'].items():
        print(f'  {"":17} {k:17} {d["lsd"]:6.3f}  {"":20} ' + ' '.join(f'{d["power_ratio"][b]:6.2f}' for b in SPEC_BANDS))
    print(f'  band error |X^-X|^2/|X|^2: {"mask":17} ' + ' '.join(f'{b:>6}' for b in SPEC_BANDS))
    for k, d in res['masked_spectrum'].items():
        print(f'  {"":26} {k:17} ' + ' '.join(f'{d["error_ratio"][b]:6.2f}' for b in SPEC_BANDS))
    print(f'  seam disagreement (unmasked, / signal power): {res["seam_disagreement"]:.4f}')
    print(f'  coord sim vs closeness {struct.get("coord_sim_vs_closeness_spearman", float("nan")):.3f}')
    if 'spatial_bias_per_block' in struct:
        print('  spatial bias per block (closeness rho / mean|b|): ' +
              ' '.join(f'{d["closeness_spearman"]:+.2f}/{d["mean_abs"]:.2f}' for d in struct['spatial_bias_per_block']))


PRESETS = {'standard'}


def add_args(flags):
    flags.add('--max-windows', type=int, default=512, help='held-out windows (atom_usage / attention_range cap it at 256)')


def compute(ctx):
    evaluate(ctx.model, ctx.config, ctx.numbers('backbone_eval.json'),
             max_windows=ctx.args.max_windows, name=ctx.config['training_params']['pretrain']['model_name'], device=ctx.device)
