"""The task filter panel (the task_filter_panel panel): a task filter (model/Qtome/Qtome_filter_tools.py) applied to one dataset it
never saw (no label, head or trial of it is used). Left: the share of power kept per Q-atom and, per Q-atom, where the
filter reads from to make the dataset's real channels (squared weight of each input channel into them). The filter
alone (importance maps, power kept of white input, effective rank) is draw_filter's figure. Right: the dataset before and after
-- the class difference of the mean waveform, the spectrum, single trials, class information in the waveform and the
log power, per-class and per-band class information maps, channel-by-frequency spectra. The dataset is encoded with
its missing channels imputed and drawn on all 64 channels (imputed ones marked *): 'before' is the reconstruction from
that encoding, 'after' the filter's output, 'removed' their difference; waveforms in microvolts.
Class information, for any number of classes: the share of variance explained by the class,
R2 = sum_k n_k (mean_k - mean)^2 / sum_i (x_i - mean)^2, per channel and time sample (phase-locked content) or per
channel and frequency of the trial's log power (power content)."""
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from IO.loader import get_standard_coords
from model.Qtome.Qtome_filter_tools import CHANNELS, FS, gains, imputed_encoder, names as channel_names, r2
from tools.viz.topomap import project_coords_2d, draw_topomap
COL = {'before': 'k', 'after': 'tab:red', 'removed': 'tab:blue'}
BANDS = {'all': (0.5, 100), 'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 100)}
INFO_CMAP = 'Greens'                                                      # class information; share of power kept is YlOrRd, spectra viridis
SMOOTH = 15                                                               # trials averaged per row of the single-trial image
CHANNEL = {'BNCI2014009': 'Pz', 'BNCI2014008': 'Fz', 'BNCI2014001': 'C3', 'BNCI2014004': 'C3', 'Nakanishi2015': 'Oz'}   # shown channel: P300 Pz (BNCI2014008: Fz, where its target effect is, d 0.30 against 0.02 at Pz), motor imagery C3, SSVEP Oz


def seam(g, backbone):
    """Disagreement of neighbouring patches on their shared half, relative to the power; gains [B, N, C, S, 2]."""
    P = backbone.patches(g); h = P.shape[-1] // 2
    return float(((P[:, :, :-1, h:] - P[:, :, 1:, :h]) ** 2).mean() / (P ** 2).mean())


def raw_trials(ds, subject, labels, T):
    """The compiled trials of one subject in microvolts, [trials, C, T] in the dataset's own channel order, checked
    against the cache's labels (same trials, same order)."""
    for f in sorted(glob.glob(f'datas/finetune/{ds}/cache/{subject}_*.npz'), key=os.path.getmtime, reverse=True):
        z = np.load(f)
        if z['data'].shape[0] == len(labels) and z['data'].shape[-1] == T and np.array_equal(z['labels'], labels):
            return z['data'].astype(np.float32) * 1e6
    raise FileNotFoundError(f'no compiled file of {ds} subject {subject} matches the cache')


def panel_numbers(ctx, ds, share, path, log=print):
    """The numbers of the panel: the filter of ctx at this share applied to dataset ds (encoded with its missing channels imputed),
    reduced to what the figure shows and saved to path (npz). Drawing is draw(path, out_path)."""
    SHARE, SRC, BACKBONE, src_c = share, ctx.filt['source'], ctx.name, CHANNELS
    md = json.load(open(f'datas/finetune/{ds}/metadata.json'))['data_metadata']; t0 = md.get('event_onset_seconds', 0.0)
    peak = ctx.backbone.freqs                                            # each Q-atom's dominant frequency
    dst_r, dst_c = channel_names(ds); encode = imputed_encoder(ctx.bb, ds, ctx.backbone.model, ctx.filt['coords64']); mean_op = ctx.filt['filters'][SHARE]   # [S, 64, 64]
    G, Y, seams, Pf, Pk, who = {'before': [], 'after': []}, [], [], 0, 0, []
    for sub, (amp, y) in sorted(gains(ctx.bb, ds, len(dst_c), None).items(), key=lambda kv: int(kv[0])):
        z = np.load(sorted(glob.glob(f'datas/finetune/{ds}/cache/{sub}_fs200*.npz'))[0]); x = z['data'].astype(np.float64)
        g = encode((x / x.std((1, 2), keepdims=True))[:, [dst_r.index(n) for n in dst_c], :int(z['valid_end'][0])], True)
        assert g.shape[:2] == amp.shape[:2], f'64-channel encoding {tuple(g.shape)} does not match the cached gains {tuple(amp.shape)}'
        kept = torch.einsum('scd,bndsx->bncsx', mean_op, g)
        G['before'].append(g); G['after'].append(kept); Y.append(y); who.append(np.full(len(y), sub))
        seams.append((seam(g[:200], ctx.backbone), seam(kept[:200], ctx.backbone)))
        Pf, Pk = Pf + g.pow(2).sum((0, -1)).numpy(), Pk + kept.pow(2).sum((0, -1)).numpy()      # [N, C, S]
    y = np.concatenate(Y); dec = lambda parts: np.concatenate([ctx.backbone.decode(g) for g in parts])
    X = {'before': dec(G['before']), 'after': dec(G['after'])}
    # back to microvolts: the model saw each trial divided by its own standard deviation over channels and time
    raw = np.concatenate([raw_trials(ds, w[0], yy, X['before'].shape[-1]) for w, yy in zip(who, Y)])
    sd = raw.std((1, 2), keepdims=True); X = {k: x * sd for k, x in X.items()}; X['removed'] = X['before'] - X['after']
    N, C, S = Pf.shape; T = X['before'].shape[-1]; t = np.arange(T) / FS - t0
    names = list(src_c)
    f = np.fft.rfftfreq(T, 1 / FS); fsel = (f >= 2) & (f <= 45); f = f[fsel]
    L = {k: np.log(np.abs(np.fft.rfft(x, axis=-1))[..., fsel] ** 2 + 1e-12) for k, x in X.items()}  # [trials, C, F]
    Re, Rp = {k: r2(x, y) for k, x in X.items()}, {k: r2(l, y) for k, l in L.items()}             # [C, T], [C, F]
    if ds in CHANNEL:                                                     # the paradigm's conventional channel; time and frequency = its peaks
        ce = cp = names.index(CHANNEL[ds]); te, fp = int(Re['before'][ce].argmax()), int(Rp['before'][cp].argmax())
    else:                                                                 # not listed: where the class information is largest
        ce, te = np.unravel_index(Re['before'].argmax(), Re['before'].shape); cp, fp = np.unravel_index(Rp['before'].argmax(), Rp['before'].shape)
    share = float(Pk.sum() / Pf.sum()); s0, s1 = np.mean(seams, 0)
    meta = json.load(open(f'datas/finetune/{ds}/metadata.json'))['data_metadata']['channels']
    rc = [v['label'] for v in meta.values() if isinstance(v, dict)].index(names[ce])
    r_ = raw[:, rc] - raw.mean((1, 2))[:, None]; b_ = X['before'][:, ce]
    uv_corr, uv_slope = float(np.corrcoef(r_.ravel(), b_.ravel())[0, 1]), float((r_ * b_).sum() / (b_ ** 2).sum())
    assert uv_corr > 0.9 and 0.8 < uv_slope < 1.25, f'microvolt conversion does not match the recording: corr {uv_corr:.2f}, slope {uv_slope:.2f}'
    log(f'{ds}: microvolts check at {names[ce]}: corr with the recording {uv_corr:.3f}, slope {uv_slope:.3f}')
    KEYS = ('before', 'after', 'removed'); AB = ('before', 'after'); dB = 10 / np.log(10)
    Ks = np.unique(y)[:4] if len(np.unique(y)) > 2 else np.unique(y)[-1:]   # band block: the first four classes; with two, one says it all
    fr = np.fft.rfftfreq(T, 1 / FS); Pw = {k: np.abs(np.fft.rfft(X[k], axis=-1)) ** 2 for k in AB}   # [trials, C, bins]
    rows_ = [('waveform', None)] + list(BANDS.items()); bm = np.zeros((len(rows_), len(Ks), 2, C)); wt = np.zeros(len(Ks))   # class information of one class against the rest
    for i, (bn, band) in enumerate(rows_):
        for ci, c in enumerate(Ks):
            if band is None:                                              # waveform: [C, T] -> the map at the class's best time (before)
                Rw = {k: r2(X[k], y == c) for k in Pw}; tc = int(Rw['before'].max(0).argmax()); wt[ci] = t[tc]
                for ki, k in enumerate(AB): bm[i, ci, ki] = Rw[k][:, tc]
            else:
                for ki, k in enumerate(AB): bm[i, ci, ki] = r2(np.log(Pw[k][..., (fr >= band[0]) & (fr < band[1])].mean(-1) + 1e-12), y == c)
    del Pw
    Kc = np.unique(y); bsel = np.abs(f - f[fp]) <= 2
    who_ = np.concatenate(who); pick = (who_ == who_[0]) & (y == Kc[-1]); ker = np.ones(SMOOTH) / SMOOTH
    img = {k: np.apply_along_axis(lambda c: np.convolve(c, ker, 'valid'), 0, X[k][pick][:, ce]) for k in AB}   # [trials, T]: one subject, one class, in recording order
    np.savez(path, ds=ds, source=SRC, backbone=BACKBONE, share_cells=SHARE, peak=np.asarray(peak), dst_c=np.array(dst_c), t=t, f=f, y=y, who=who_,
             n_sub=len(Y), npick=int(pick.sum()), ce=ce, cp=cp, te=te, fp=fp, kept=share, seam=np.array([s0, s1]),
             Re=np.stack([Re[k] for k in KEYS]), Rp=np.stack([Rp[k] for k in KEYS]), ps=np.stack([dB * L[k].mean(0) for k in KEYS]),
             cd=np.stack([np.stack([X[k][y == c][:, ce].mean(0) - X[k][:, ce].mean(0) for c in Kc]) for k in KEYS]),
             dif=np.stack([np.stack([L[k][y == c][:, cp].mean(0) - L[k][:, cp].mean(0) for c in Kc]) for k in AB]),
             tp=np.stack([np.stack([L[k][y == c][:, :, bsel].mean((0, 2)) - L[k][:, :, bsel].mean((0, 2)) for k in AB]) for c in Kc[:4]]),
             img=np.stack([img[k] for k in AB]), avg=np.stack([X[k][pick][:, ce].mean(0) for k in AB]), bm=bm, wt=wt,
             ka=Pk.sum((0, 1)) / Pf.sum((0, 1)), reads=mean_op[:, [names.index(n) for n in dst_c]].pow(2).sum(1).T.numpy())   # [S]; [C, S] squared weight of each input channel into the real channels


def draw(path, out_path, log=print):
    """Draw the panel from panel_numbers' file: the filter at one share applied to one dataset."""
    z = np.load(path); ds, SRC, BACKBONE, SHARE = str(z['ds']), str(z['source']), str(z['backbone']), float(z['share_cells'])
    md = json.load(open(f'datas/finetune/{ds}/metadata.json'))['data_metadata']
    cls = lambda c: md.get('targets', {}).get(str(int(c)), {}).get('label', f'class {c}')        # the dataset's own class names
    KEYS = ('before', 'after', 'removed')
    peak, dst_c, names = z['peak'], [str(n) for n in z['dst_c']], list(CHANNELS); C, S = len(names), len(peak)
    xyz = np.array([get_standard_coords(n) for n in names]); pos = project_coords_2d(xyz)   # every channel, at its standard position
    t, f, y, who_, n_sub, npick = z['t'], z['f'], z['y'], z['who'], int(z['n_sub']), int(z['npick'])
    ce, cp, te, fp = (int(z[k]) for k in ('ce', 'cp', 'te', 'fp')); share, (s0, s1) = float(z['kept']), z['seam']
    Re, Rp, ps, cd = (dict(zip(KEYS, z[k])) for k in ('Re', 'Rp', 'ps', 'cd'))
    dif, img, avg = (dict(zip(('before', 'after'), z[k])) for k in ('dif', 'img', 'avg')); ka, reads, bm, wt = z['ka'], z['reads'], z['bm'], z['wt']
    Ks = np.unique(y)[:4] if len(np.unique(y)) > 2 else np.unique(y)[-1:]   # band block: the first four classes; with two, one says it all
    fig = plt.figure(figsize=(37.5 + 5.2 * len(Ks), 27))                                    # fixed grids, no layout engine: it shrank the plots
    # left: the filter itself. Right: the filter applied to the dataset -- signal plots, then the two class-analysis blocks
    outer = fig.add_gridspec(1, 4, width_ratios=[10, 21, 6.5, 5.2 * len(Ks)], left=.012, right=.988, top=.925, bottom=.035, wspace=.07)
    FILT, MAIN, CLS, BAND = outer[0], outer[1], outer[2], outer[3]; fl = FILT.subgridspec(2, 1, height_ratios=[1.1, 6.2], hspace=.22)
    gs = MAIN.subgridspec(3, 4, height_ratios=[1, 1, 1.5], hspace=.35, wspace=.28)
    ax = np.array([[fig.add_subplot(gs[i - 1, j]) if i > 1 or (i == 1 and j > 1) else None for j in range(4)] for i in range(3)], dtype=object)
    gside = fl[1].subgridspec(6, 4, hspace=.32, wspace=.35)
    # middle block: class information of ONE class against all the others (column pairs before | after), per channel:
    # first row in the waveform, at the time where that class is most separable; then in the log power of each band
    gband = BAND.subgridspec(len(BANDS) + 1, 2 * len(Ks), hspace=.25, wspace=.08)
    rows_ = [('waveform', None)] + list(BANDS.items())
    for i, (bn, band) in enumerate(rows_):
        dd = {(c, k): bm[i, ci, ki] for ci, c in enumerate(Ks) for ki, k in enumerate(('before', 'after'))}      # [C] each
        note = {c: f'{wt[ci]:.2f} s' for ci, c in enumerate(Ks)}
        vb = max(d.max() for d in dd.values()); row = []
        for j, (c, k) in enumerate(dd):
            a_ = fig.add_subplot(gband[i, j]); im = draw_topomap(a_, pos, dd[(c, k)], vmin=0, vmax=vb, cmap=INFO_CMAP); row.append(a_)
            a_.set_title((f'{cls(c)}\n' if i == 0 else '') + k + (f', {note[c]}' if band is None else ''), fontsize=9)
            if j == 0: a_.text(-.12, .5, bn if band is None else f'{bn}\n{band[0]:g}-{band[1]:g} Hz', transform=a_.transAxes, rotation=90, ha='center', va='center', fontsize=11)
        fig.colorbar(im, ax=row, shrink=.7, fraction=.02, pad=.01).ax.tick_params(labelsize=7)
    xm = BAND.get_position(fig); fig.text((xm.x0 + xm.x1) / 2, .96, 'class information of each class\nagainst the others: waveform,\nthen log power per band (own scale per row)', ha='center', va='top', fontsize=13)
    # left: how much of each Q-atom's power is kept on this dataset, and where on the scalp it reads from
    sa = [fig.add_subplot(gside[i, j]) for i in range(6) for j in range(4)]
    for i, a_ in zip(np.argsort(-ka), sa):                                # most kept first, as the bars
        im = draw_topomap(a_, pos, reads[:, i], cmap='Purples'); a_.set_title(f'{i}: {peak[i]:.0f} Hz, kept {ka[i]:.2f}', fontsize=8)
        fig.colorbar(im, ax=a_, shrink=.75, pad=.03, fraction=.06).ax.tick_params(labelsize=5)
    xs_ = gside.get_grid_positions(fig); fig.text((xs_[2][0] + xs_[3][-1]) / 2, xs_[1][0] + .018, f'where each Q-atom reads from (index: Hz, power kept), most kept first\nsquared weight of each input channel into {", ".join(dst_c)}; own scale per map', ha='center', va='bottom', fontsize=11)
    # row 0: what the time-free mask keeps of each Q-atom's power, most kept first, after the total
    hm = fig.add_subplot(fl[0]); order = np.argsort(-ka); vals = np.r_[share, ka[order]]; xb = np.r_[-0.5, 1 + np.arange(S)]
    hm.bar(xb, vals, color=plt.cm.YlOrRd(vals / vals.max()), edgecolor='k', lw=.6)
    for x_, v_ in zip(xb, vals): hm.text(x_, v_ + .01 * vals.max(), f'{v_:.2f}', ha='center', va='bottom', fontsize=7)
    hm.axvline(.25, color='k', lw=1.5); hm.set_xlim(-1.2, S + .6); hm.set_ylim(0, 1.1 * vals.max()); hm.grid(axis='y', alpha=.3)
    hm.set_xticks(xb); hm.set_xticklabels(['all'] + [f'{i}\n{peak[i]:.0f}' for i in order], fontsize=7); hm.set_ylabel('share of power kept')
    hm.set_title('share of power kept per Q-atom (index, Hz), most kept first', fontsize=11)
    xf, xr = FILT.get_position(fig), MAIN.get_position(fig); xsep = (xf.x1 + xr.x0) / 2
    fig.add_artist(plt.Line2D([xsep, xsep], [.02, .985], color='k', lw=2))
    fig.text((xf.x0 + xf.x1) / 2, .992, f'THE FILTER ON {ds}', ha='center', va='top', fontsize=20, weight='bold')
    fig.text((xr.x0 + .988) / 2, .992, f'APPLIED TO {ds} (imputed to 64 channels): before and after', ha='center', va='top', fontsize=22, weight='bold')
    # row 1: the class difference of the mean waveform (wide) and the power spectrum at the shown channel, in dB
    for k in KEYS:
        ax[1, 2].plot(f, ps[k][ce], COL[k], label=k)
    ax[1, 2].set_title(f'power spectrum at {names[ce]}'); ax[1, 2].legend(); ax[1, 2].set_xlabel('Hz'); ax[1, 2].set_ylabel('power, dB re 1 µV²'); ax[1, 2].grid(alpha=.3)
    ax[1, 3].axis('off')
    ax[1, 3].text(0, 1, '\n'.join([
        f'{ds}, {BACKBONE}', f'{n_sub} subjects, {len(y)} trials, {len(np.unique(y))} classes:',
        *[f'  class {c} = {cls(c)} ({int((y == c).sum())} trials)' for c in np.unique(y)], '',
        f'filter: component cells, share {SHARE:g}, class information less chance',
        f'  FITTED ON {SRC}, which this dataset never met', '  applied to all 64 channels, missing ones imputed',
        '  time-free: averaged over the patches',
        f'  factor 1 for the top {SHARE:.0%} of cells, 0 for the rest', '',
        f'power kept: {share:.0%}', f'seam error: {s0:.3f} -> {s1:.3f}', '',
        'class information R2 = share of variance', 'explained by the class',
        f'  phase-locked, peak at {names[ce]}: {Re["before"][ce].max():.3f} -> {Re["after"][ce].max():.3f}', f'    removed part: {Re["removed"][ce].max():.3f}',
        f'  log power, peak at {names[cp]}: {Rp["before"][cp].max():.3f} -> {Rp["after"][cp].max():.3f}', f'    removed part: {Rp["removed"][cp].max():.3f}']),
        va='top', family='monospace', fontsize=10.5)
    # each class's mean log power minus the mean of all trials, at the shown channel (a channel average cancels
    # what differs in opposite directions on the two sides of the head, as left and right hand imagery do)
    Kc = np.unique(y); cc = plt.cm.turbo(np.linspace(0.05, 0.95, len(Kc)))
    v = 1.05 * max(np.abs(d).max() for d in dif.values())
    # far-left block: the same quantity as the lines of row 2 on every channel, in the band around the frequency where
    # the classes differ most at the shown channel -- one row per class, before | after, one shared scale
    kt = Kc[:4]; tg = CLS.subgridspec(len(kt), 2, hspace=.2, wspace=.05)
    tp_ = {(c, k): z['tp'][ci, ki] for ci, c in enumerate(kt) for ki, k in enumerate(('before', 'after'))}   # [C]
    vt = max(np.abs(m_).max() for m_ in tp_.values()); maps = []
    for i_, c in enumerate(kt):
        for j_, k in enumerate(('before', 'after')):
            a_ = fig.add_subplot(tg[i_, j_]); im = draw_topomap(a_, pos, tp_[(c, k)], vmin=-vt, vmax=vt, cmap='RdBu_r'); maps.append(a_)
            a_.set_title(f'{cls(c)}: {k}', fontsize=12)
    fig.colorbar(im, ax=maps, shrink=.5, fraction=.03, pad=.02, location='bottom', label='difference in log power').ax.tick_params(labelsize=8)
    xl_ = CLS.get_position(fig); fig.text((xl_.x0 + xl_.x1) / 2, .96, f'class mean minus mean of all trials\nlog power at {f[fp]:.0f} +- 2 Hz, per channel', ha='center', va='top', fontsize=13)
    for j, k in enumerate(('before', 'after')):                           # row 2, left pair: before | after
        for d, c, col in zip(dif[k], Kc, cc):
            ax[2, j].plot(f, d, color=col, lw=1.3, label=cls(c))
        ax[2, j].axhline(0, color='gray', lw=.5); ax[2, j].set_ylim(-v, v); ax[2, j].set_xlabel('Hz')
        ax[2, j].set_title(f'{k}: class spectrum minus mean spectrum\n(log power at {names[cp]})')
    ax[2, 0].legend(fontsize=7, ncol=2 if len(Kc) > 4 else 1)
    # class information, phase-locked (time) and power (frequency): curves in row 2 (right pair); the head maps are in the per-class block
    for i, (R, xs, p0, xl, what) in enumerate(((Re, t, te, 's from event', 'waveform'), (Rp, f, fp, 'Hz', 'log power'))):
        for k in R:
            ax[2, 2 + i].plot(xs, R[k].max(0), COL[k], label=k)
        ax[2, 2 + i].set_title(f'class information in the {what}\n(R2, largest over channels)'); ax[2, 2 + i].legend(); ax[2, 2 + i].set_xlabel(xl)
    # bottom row: single trials over the recording, and the spectra by channel
    # single trials of one subject and one class in recording order (rows smoothed over SMOOTH trials): does the response move?
    v = np.abs(img['before']).max(); va = 1.1 * max(np.abs(a).max() for a in avg.values())
    for j, k in enumerate(('before', 'after')):                           # the image, and under it the mean of the same trials
        sub = gs[2, j].subgridspec(2, 1, height_ratios=[3, 1], hspace=.06); a_, b_ = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        im = a_.imshow(img[k], aspect='auto', origin='lower', cmap='RdBu_r', vmin=-v, vmax=v, extent=[t[0], t[-1], 0, len(img[k])])
        fig.colorbar(im, ax=[a_, b_], fraction=.05, pad=.02)
        a_.axvline(0, color='k', lw=.6); a_.set_xticklabels([]); a_.set_ylabel('trial, in recording order')
        fig.axes[-1].set_ylabel('µV')
        a_.set_title(f'{k}: single trials at {names[ce]}, subject {who_[0]}, {cls(Kc[-1])}\n({npick} trials, mean of {SMOOTH} neighbours; below: mean of all)')
        b_.plot(t, avg[k], COL[k], lw=1.5); b_.axhline(0, color='gray', lw=.5); b_.axvline(0, color='k', lw=.6)
        b_.set_xlim(t[0], t[-1]); b_.set_ylim(-va, va); b_.set_xlabel('s from event'); b_.set_ylabel('mean, µV')
    a_ = fig.add_subplot(gs[0, :2])
    for k in KEYS:
        a_.plot(t, cd[k][-1] - cd[k][0] if len(Kc) == 2 else np.sqrt((cd[k] ** 2).mean(0)), COL[k], lw=1.5, label=k)
    a_.axhline(0, color='gray', lw=.5); a_.axvline(0, color='k', lw=.6); a_.legend(); a_.set_xlabel('s from event'); a_.set_ylabel('µV')
    a_.set_title(f'class difference of the mean waveform at {names[ce]}: ' + (f'{cls(Kc[-1])} minus {cls(Kc[0])}' if len(Kc) == 2 else 'size over the classes (rms)'))
    front = np.argsort(-xyz[:, 1], kind='stable')                         # channels front to back
    lo, hi = min(ps['before'].min(), ps['after'].min()), ps['before'].max()
    sub = gs[2, 2:].subgridspec(1, 3, wspace=.5); dif_ = ps['after'] - ps['before']; vd = np.abs(dif_).max()
    for j, (k, M_, kw) in enumerate((('before', ps['before'], dict(cmap='viridis', vmin=lo, vmax=hi)), ('after', ps['after'], dict(cmap='viridis', vmin=lo, vmax=hi)),
                                     ('after minus before', dif_, dict(cmap='RdBu_r', vmin=-vd, vmax=vd)))):
        a_ = fig.add_subplot(sub[j]); im = a_.imshow(M_[front], aspect='auto', origin='upper', extent=[f[0], f[-1], C - .5, -.5], **kw)
        fig.colorbar(im, ax=a_, fraction=.06, pad=.03, label='dB re 1 µV²' if j < 2 else 'dB')
        a_.set_yticks(range(C)); a_.set_yticklabels([names[c] + ('' if names[c] in dst_c else '*') for c in front], fontsize=8 if C < 30 else 5); a_.set_xlabel('Hz')
        a_.set_title(f'{k}: power spectrum\nchannel (front to back) by frequency', fontsize=11)
    fig.savefig(out_path, dpi=80); plt.close(fig)
    log(f'{ds}, filter from {SRC}, share {SHARE:g}: power kept {share:.0%}, seam {s0:.3f} -> {s1:.3f}; '
        f'R2 phase-locked {Re["before"][ce].max():.3f} -> {Re["after"][ce].max():.3f} (removed {Re["removed"][ce].max():.3f}) at {names[ce]} {t[te]:.2f} s; '
        f'log power {Rp["before"][cp].max():.3f} -> {Rp["after"][cp].max():.3f} (removed {Rp["removed"][cp].max():.3f}) at {names[cp]} {f[fp]:.1f} Hz')


def filter_numbers(ctx, share, path):
    """The numbers of the filter alone, saved to path (npz): per Q-atom dominant frequency, power kept of white input and
    effective rank of the filter at this share, and the importance [64, S] of each channel for each Q-atom."""
    mean_op = ctx.filt['filters'][share].numpy(); C = mean_op.shape[1]
    sv = np.linalg.svd(mean_op, compute_uv=False)
    np.savez(path, peak=np.asarray(ctx.backbone.freqs), white=(sv ** 2).sum(1) / C, rank=sv.sum(1) ** 2 / (sv ** 2).sum(1),
             importance=ctx.filt['importance'].numpy(), source=ctx.filt['source'], share=share)


def draw_filter(path, out_path, log=print):
    """The filter alone, whatever it is applied to, drawn from filter_numbers' file: per Q-atom the power it keeps of white
    input (equal power on every channel, no correlation), its effective rank (channel patterns kept), and IMPORTANCE of each
    channel (class information less chance from the heads that built it, as a share of the total over every channel and
    Q-atom; ONE colour scale, so a Q-atom that matters little is pale everywhere). Q-atoms most important first."""
    z = np.load(path); SRC, SHARE, peak, white, rank, imp = str(z['source']), float(z['share']), z['peak'], z['white'], z['rank'], z['importance']
    C, S = imp.shape; imp = imp / imp.sum(); ia = imp.sum(0); order = np.argsort(-ia)
    pos = project_coords_2d(np.array([get_standard_coords(n) for n in CHANNELS]))
    fig = plt.figure(figsize=(24, 27)); fl = fig.add_gridspec(2, 1, height_ratios=[.9, 6.2], left=.02, right=.98, top=.93, bottom=.035, hspace=.12)
    fo = fl[0].subgridspec(1, 2, wspace=.2); a_ = fig.add_subplot(fo[0]); a_.scatter(peak, white, c=white, cmap='YlOrRd', edgecolor='k', s=70, zorder=3)
    for i in range(S): a_.annotate(str(i), (peak[i], white[i]), fontsize=8, xytext=(4, 3), textcoords='offset points')
    a_.set_xscale('log'); a_.set_xticks([2, 4, 8, 13, 30, 60, 100]); a_.set_xticklabels([2, 4, 8, 13, 30, 60, 100]); a_.grid(alpha=.3); a_.set_ylim(0, None)
    a_.set_xlabel('dominant frequency of the Q-atom, Hz'); a_.set_ylabel('power kept'); a_.set_title('power kept of white input, per Q-atom')
    fq = np.argsort(peak, kind='stable')                                  # by frequency, as the scatter
    a_ = fig.add_subplot(fo[1]); a_.bar(np.arange(S), rank[fq], color=plt.cm.YlOrRd(ia[fq] / ia.max()), edgecolor='k', lw=.6)
    a_.set_xticks(np.arange(S)); a_.set_xticklabels([f'{i}\n{peak[i]:.0f}' for i in fq], fontsize=8); a_.set_ylim(0, C); a_.grid(axis='y', alpha=.3)
    a_.set_xlabel('Q-atom (index, Hz), by frequency; colour: importance'); a_.set_ylabel(f'of {C} channels'); a_.set_title('number of channel patterns kept (effective rank)')
    gm = fl[1].subgridspec(6, 4, hspace=.32, wspace=.2); sa = [fig.add_subplot(gm[i, j]) for i in range(6) for j in range(4)]
    for i, a_ in zip(order, sa):
        im = draw_topomap(a_, pos, imp[:, i], vmin=0, vmax=imp.max(), cmap='YlOrRd'); a_.set_title(f'{i}: {peak[i]:.0f} Hz, {ia[i]:.1%}', fontsize=9)
    fig.colorbar(im, ax=sa, shrink=.4, fraction=.02, pad=.01, location='bottom', label='share of the total importance').ax.tick_params(labelsize=8)
    fig.text(.5, .99, f'THE FILTER fitted on {SRC}, all 64 channels, share {SHARE:g}, time-free', ha='center', va='top', fontsize=20, weight='bold')
    xs_ = gm.get_grid_positions(fig); fig.text(.5, xs_[1][0] + .012, 'importance of each channel for each Q-atom: class information less chance (index: Hz, share of the total), most important first; one scale', ha='center', va='bottom', fontsize=13)
    fig.savefig(out_path, dpi=80); plt.close(fig)
    log(f'filter from {SRC}, share {SHARE:g}: {out_path}')
