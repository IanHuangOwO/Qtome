"""Tests of a task filter (model/Qtome/Qtome_filter_tools.py) on a dataset it never saw, run by task_filter.py eval. The target
is encoded with its missing channels imputed, the whole 64-channel filter is applied and the target's real channels
are read out. Every test compares the filter with 'matched q': the 64-channel reconstruction filtered (zero phase) to
the filter output's mean power spectrum, so the same frequencies are removed with no channel or Q-atom choice.
  mi     motor-imagery target. Signals: raw; band 8-30 Hz; backbone (the reconstruction from the cached gains of the
         real channels); backbone64 (from the 64-channel encoding, no filter: what the imputation alone changes);
         full64 q; matched q. Scores: change (relative RMS difference to raw), kept (power against backbone), effect
         (Cohen's d, left against right hand, log 8-30 Hz power C3 minus C4), lda (balanced accuracy of a shrinkage LDA
         on log 8-13 and 13-30 Hz power of every channel, 5 folds in trial order, all classes).
  p300   P300 target. Signals: raw; low-pass 12 Hz; backbone64; full64 q; matched q. Scores: kept (against
         backbone64), reliability (split-half correlation of the target average, 30 + 30 trials, 100 draws),
         difference (the same for target minus non-target), lda (ROC AUC of a shrinkage LDA on all channels x 50 ms
         means, 5 folds in trial order).
  noise  motor-imagery target with one artifact added to every trial at POWER of the trial's power: 'noisy channel'
         (white noise on one channel, drawn per trial), 'blink' (slow frontal wave), 'muscle' (20-90 Hz burst over one
         temporal side). Signals: clean; input; backbone64; full64 q; matched q; band 8-30 Hz; ASR (asrpy, cutoff 20,
         calibrated on the subject's clean trials: a favour, real use has no clean copy). Scores: signal and artifact
         kept and their ratio in dB (the cleaned trial c regressed on the clean trial x and the noise n, c ~ s*x + a*n),
         effect and lda as in mi.
Per subject, then mean +- SE over subjects; paired differences with their own standard error."""
import glob

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from model.Qtome.Qtome_filter_tools import CHANNELS, FS, MONT, gains, imputed_encoder, names

POWER = 0.5                                                               # noise: artifact power as a share of the trial's
KINDS = ('noisy channel', 'blink', 'muscle')
FPZ, T7, T8 = (0.00011, 0.08825, -0.00171), (-0.08416, -0.01602, -0.00935), (0.08508, -0.01502, -0.00949)


def band_power(x, lo=8, hi=30):                                           # [T, C, time] -> log power [T, C]
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS); P = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    return np.log(P[..., (f >= lo) & (f <= hi)].mean(-1) + 1e-12)


def cohen_d(a, b):
    return float((a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))


def lda(w, y):                                                            # motor imagery
    X = np.concatenate([band_power(w, 8, 13), band_power(w, 13, 30)], 1)
    return float(balanced_accuracy_score(y, cross_val_predict(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), X, y, cv=StratifiedKFold(5))))


def lda_auc(w, y):                                                        # P300
    X = w[..., :w.shape[-1] // 10 * 10].reshape(*w.shape[:2], -1, 10).mean(-1).reshape(len(w), -1)   # splits the time axis only, then joins channel x time
    return float(roc_auc_score(y, cross_val_predict(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), X, y, cv=StratifiedKFold(5), method='decision_function')))


def reliability(E, n=30, draws=100):                                      # [trials, C, T] -> mean correlation of two halves' averages
    g = np.random.default_rng(0)                                          # the same draws for every signal
    return float(np.mean([np.corrcoef(E[p[:n]].mean(0).ravel(), E[p[n:2 * n]].mean(0).ravel())[0, 1] for p in (g.permutation(len(E)) for _ in range(draws))]))


def difference(w, y, n=30, draws=100):                                    # split-half reliability of target minus non-target
    g = np.random.default_rng(0); t, o = np.flatnonzero(y == 1), np.flatnonzero(y == 0); r = []
    for _ in range(draws):
        a, b = g.permutation(t), g.permutation(o); d = lambda i: (w[a[i * n:(i + 1) * n]].mean(0) - w[b[i * n:(i + 1) * n]].mean(0)).ravel()
        r.append(np.corrcoef(d(0), d(1))[0, 1])
    return float(np.mean(r))


def matched(full, masked):                                                # full filtered (zero phase) to masked's mean power spectrum
    F = np.fft.rfft(full, axis=-1); g = np.sqrt((np.abs(np.fft.rfft(masked, axis=-1)) ** 2).mean((0, 1)) / ((np.abs(F) ** 2).mean((0, 1)) + 1e-20))
    return np.fft.irfft(F * g, full.shape[-1], axis=-1)


def kept(c, x, n):                                                        # c ~ s*x + a*n over everything -> s, a
    s, a = np.linalg.lstsq(np.stack([x.ravel(), n.ravel()], 1).astype(np.float64), c.ravel().astype(np.float64), rcond=None)[0]
    return float(s), float(a)


def hann_at(T, centre, width):                                            # [T] bump of `width` samples
    t = np.arange(T) - centre
    return np.where(np.abs(t) < width / 2, 0.5 + 0.5 * np.cos(2 * np.pi * t / width), 0.0)


def artifacts(B, coords, T, rng):
    """{name: n [B, C, T]}: shapes only, scaled later to POWER times the trial's power."""
    C = len(coords)
    near = lambda p, tau: np.exp(-np.linalg.norm(coords - np.array(p), axis=1) / tau)     # [C] falls off with distance
    out = {'blink': near(FPZ, 0.06)[None, :, None] * np.stack([hann_at(T, rng.integers(T // 4, 3 * T // 4), 0.4 * FS) for _ in range(B)])[:, None]}
    bad = np.zeros((B, C, T)); bad[np.arange(B), rng.integers(0, C, B)] = rng.standard_normal((B, T))
    out['noisy channel'] = bad
    f = np.fft.rfftfreq(T, 1 / FS); band = (f >= 20) & (f <= 90)
    emg = np.fft.irfft(np.fft.rfft(rng.standard_normal((B, 1, T))) * band, T)
    side = np.where(rng.random(B)[:, None] < 0.5, near(T7, 0.04)[None], near(T8, 0.04)[None])   # [B, C]
    env = np.stack([hann_at(T, rng.integers(T // 4, 3 * T // 4), min(T, FS)) for _ in range(B)])[:, None]
    out['muscle'] = side[:, :, None] * emg * env
    return {k: v.astype(np.float32) for k, v in out.items()}


def asr_clean(noisy, calib, chans):
    import asrpy, mne                                                     # only this test needs them
    mne.set_log_level('ERROR')
    raw = lambda w: mne.io.RawArray(np.concatenate(list(w), -1).astype(np.float64), mne.create_info(chans, FS, 'eeg')).set_montage('standard_1005')
    asr = asrpy.ASR(sfreq=FS, cutoff=20); asr.fit(raw(calib))
    return np.stack(np.split(asr.transform(raw(noisy)).get_data(), len(noisy), -1)).astype(np.float32)


def subjects(ds):                                                         # compiled trials per subject, numeric order
    return sorted(glob.glob(f'datas/finetune/{ds}/cache/*_fs200*.npz'), key=lambda p: int(p.split('/')[-1].split('_')[0]))


def table(rows, title, pairs, log):
    """Print mean +- SE over subjects per signal, then the paired differences; rows {signal: {score: [per subject]}}."""
    ref = next(iter(rows.values())); n = len(next(iter(ref.values()))); keys = list(ref)
    ms = lambda v: f'{np.mean(v):8.3f} +-{np.std(v, ddof=1) / np.sqrt(n):5.3f}'
    log(f'\n{title}: {n} subjects, mean +- SE over subjects')
    log('  ' + ' ' * 28 + ''.join(f'{k:>16s}' for k in keys))
    for m, r in rows.items(): log(f'  {m:28s}' + ''.join(ms(r[k]) for k in keys))
    for a, b in pairs:
        log(f'  {a + " - " + b:28s}' + ''.join(ms(np.array(rows[a][k]) - np.array(rows[b][k])) for k in keys))


def _setup(ctx, dst):
    dst_r, dst_c = names(dst); sel = [CHANNELS.index(n) for n in dst_c]
    encode = imputed_encoder(ctx.bb, dst, ctx.backbone.model, ctx.filt['coords64'])
    return dst_r, dst_c, sel, encode, {q: o[:, sel] for q, o in ctx.filt['filters'].items()}   # [S, C_dst, 64]


def mi(ctx, dst, log=print):
    dst_r, dst_c, sel, encode, ops = _setup(ctx, dst); dec = ctx.backbone.decode; rows = {}
    for sub, (amp, y) in gains(ctx.bb, dst, len(dst_c), None).items():
        z = np.load(sorted(glob.glob(f'datas/finetune/{dst}/cache/{sub}_fs200*.npz'))[0]); assert np.array_equal(z['labels'], y)
        x = z['data'].astype(np.float64); x = x / x.std((1, 2), keepdims=True)
        back = dec(amp); T = min(back.shape[-1], int(z['valid_end'][0])); raw = x[:, [dst_r.index(n) for n in dst_c], :T]
        e = encode(raw[:8], False); c_ = np.corrcoef(e.numpy().ravel(), amp[:8].numpy().ravel())[0, 1]   # this encoder gives the cached gains
        assert e.shape == amp[:8].shape and c_ > 0.999, f'encoder does not give the cached gains: {tuple(e.shape)} against {tuple(amp[:8].shape)}, correlation {c_:.4f}'
        g64 = encode(raw, True); b64 = dec(g64[:, :, sel])[..., :T]
        sig = {'raw': raw, 'band 8-30 Hz': sosfiltfilt(butter(4, [8, 30], btype='band', fs=FS, output='sos'), raw, axis=-1),
               'backbone': back[..., :T], 'backbone64': b64}
        for q, o in ops.items():
            sig[f'full64 {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', o, g64))[..., :T]
            sig[f'matched {q:g}'] = matched(b64, sig[f'full64 {q:g}'])
        for m, w in sig.items():
            bp = band_power(w); lat = bp[:, dst_c.index('C3')] - bp[:, dst_c.index('C4')]
            for k, v in (('change', np.sqrt(((w - raw) ** 2).sum() / (raw ** 2).sum())), ('kept', (w ** 2).sum() / (sig['backbone'] ** 2).sum()),
                         ('effect', cohen_d(lat[y == 0], lat[y == 1])), ('lda', lda(w, y))):
                rows.setdefault(m, {}).setdefault(k, []).append(float(v))
        log(f'  {dst} subject {sub} done')
    table(rows, f'filter from {ctx.filt["source"]} on {dst} ({len(dst_c)} channels, missing ones imputed)',
          [('backbone64', 'backbone')] + [(f'full64 {q:g}', o.format(q=q)) for q in ops for o in ('matched {q:g}', 'band 8-30 Hz')], log)
    return rows


def p300(ctx, dst, log=print):
    dst_r, dst_c, sel, encode, ops = _setup(ctx, dst); dec = ctx.backbone.decode; rows = {}
    for f in subjects(dst):
        z = np.load(f); y = z['labels']; x = z['data'].astype(np.float64); x = x / x.std((1, 2), keepdims=True)
        T = int(z['valid_end'][0]); x = x[:, [dst_r.index(n) for n in dst_c], :T]
        g = encode(x, True); sig = {'raw': x, 'low-pass 12 Hz': sosfiltfilt(butter(4, 12, fs=FS, output='sos'), x, axis=-1),
                                    'backbone64': dec(g[:, :, sel])[..., :T]}
        for q, o in ops.items():
            sig[f'full64 {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', o, g))[..., :T]
            sig[f'matched {q:g}'] = matched(sig['backbone64'], sig[f'full64 {q:g}'])
        for m, w in sig.items():
            for k, v in (('kept', (w ** 2).sum() / (sig['backbone64'] ** 2).sum()), ('reliability', reliability(w[y == 1])),
                         ('difference', difference(w, y)), ('lda', lda_auc(w, y))):
                rows.setdefault(m, {}).setdefault(k, []).append(float(v))
        log(f'  {dst} {f.split("/")[-1].split("_")[0]} done')
    table(rows, f'filter from {ctx.filt["source"]} on {dst} ({len(dst_c)} channels, missing ones imputed)',
          [(f'full64 {q:g}', o.format(q=q)) for q in ops for o in ('matched {q:g}', 'low-pass 12 Hz', 'raw')], log)
    return rows


def noise(ctx, dst, seed=0, log=print):
    dst_r, dst_c, sel, encode, ops = _setup(ctx, dst); dec = ctx.backbone.decode; rng = np.random.default_rng(seed)
    lab = {m['label']: [m['cartesian'][k] for k in 'xyz'] for m in MONT}; xyz = np.array([lab[n] for n in dst_c])
    band = lambda w: sosfiltfilt(butter(4, [8, 30], btype='band', fs=FS, output='sos'), w, axis=-1)
    rows = {k: {} for k in KINDS}
    for f in subjects(dst):
        z = np.load(f); y = z['labels']; x = z['data'].astype(np.float64); x = x / x.std((1, 2), keepdims=True)
        T = int(z['valid_end'][0]); x = x[:, [dst_r.index(n) for n in dst_c], :T]
        for kind in KINDS:
            n = artifacts(len(x), xyz, T, rng)[kind].astype(np.float64)
            n = n * np.sqrt(POWER * (x ** 2).sum((1, 2), keepdims=True) / (n ** 2).sum((1, 2), keepdims=True)); w1 = x + n
            g = encode(w1, True); sig = {'clean': x, 'input': w1, 'backbone64': dec(g[:, :, sel])[..., :T]}
            for q, o in ops.items():
                sig[f'full64 {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', o, g))[..., :T]
                sig[f'matched {q:g}'] = matched(sig['backbone64'], sig[f'full64 {q:g}'])
            sig['band 8-30 Hz'] = band(w1)
            sig['ASR'] = asr_clean(w1.astype(np.float32) * 1e-5, x.astype(np.float32) * 1e-5, dst_c).astype(np.float64) * 1e5   # ~10 microvolts for mne
            for m, w in sig.items():
                s_, a_ = kept(w, x, n); bp = band_power(w); l_ = bp[:, dst_c.index('C3')] - bp[:, dst_c.index('C4')]
                for k, v in (('signal', s_), ('artifact', a_), ('gain dB', 10 * np.log10(max(s_, 1e-3) / max(a_, 1e-3))),
                             ('effect', cohen_d(l_[y == 0], l_[y == 1])), ('lda', lda(w, y))):
                    rows[kind].setdefault(m, {}).setdefault(k, []).append(float(v))
        log(f'  {dst} {f.split("/")[-1].split("_")[0]} done')
    for kind, r in rows.items():
        table(r, f'{kind} at {POWER:.0%} of trial power on {dst}, filter from {ctx.filt["source"]}',
              [(f'full64 {q:g}', o.format(q=q)) for q in ops for o in ('matched {q:g}', 'band 8-30 Hz', 'ASR', 'input')], log)
    return rows


TESTS = {'mi': mi, 'p300': p300, 'noise': noise}
DEFAULT_TARGETS = {'mi': ['BNCI2014001', 'BNCI2014004'], 'p300': ['BNCI2014008'], 'noise': ['BNCI2014004']}


def _selfcheck():
    g = np.random.default_rng(1); x, n = g.standard_normal((2, 4, 3, 50))
    assert np.allclose(kept(0.7 * x + 0.2 * n, x, n), (0.7, 0.2), atol=1e-6)
    a = g.standard_normal(4000) + 0.5; assert abs(cohen_d(a, g.standard_normal(4000)) - 0.5) < 0.1
    nz = artifacts(8, g.standard_normal((16, 3)) * 0.08, 200, g)
    assert all(v.shape == (8, 16, 200) for v in nz.values()) and ((nz['noisy channel'] ** 2).sum(-1) > 0).sum(1).tolist() == [1] * 8
