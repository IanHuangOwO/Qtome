"""task_filter_noise_removal_synthetic: the task filter (--filter-source) on a clean dataset with artificial noise added at the strength TUAR's real artifacts have ->
pretrain/analysis/filter/<source>/synthetic_<dataset>.npz, numbers only (the figure is drawn with --redraw; the group panel noise_removal_synthetic draws the all-seed one). Builds the filter first if it is not saved.

Why: TUAR has no clean copy of an artifact window, so its scores are power ratios against a clean-window control. Here the clean signal x and the
added noise n are both known, so a method's output c is regressed on them (c ~ s*x + a*n over the windows): s = signal kept, a = noise left
(1 = untouched). The same power-ratio scores as task_filter_artifacts are reported too, each with the control (the method on the clean windows).
Noise: eye (frontal blink wave + a slow step), muscle (30-90 Hz burst, one temporal side plus a diffuse part), electrode (a pop that decays over
0.3 s on one channel), line (60 Hz on every channel). Strength --syn-levels: dB the scored quantity (frontal 1-10 Hz power / 30-90 Hz power /
squared 99.9th-percentile amplitude / 59-61 Hz power) rises over the window's own clean value; TUAR's real artifacts are at 1 / 3 / 5 dB over the
patient's clean median (25th / 50th / 75th percentile over its annotated windows), the default. ASR is calibrated on the clean copies (a favour).
Methods are those of task_filter_artifacts (atom removal needs the atom_labels panel first). Numbers per subject, mean +- SE over subjects."""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from panels.core.numbers import load_npz, save_npz
from panels.pretrain.task_filter_artifacts import ATOM_COLORS, ATOM_GROUPS, HIGH, LINE, SLOW
from panels.shared.task_filter.cells import FS, MONT
from panels.shared.task_filter.atom_asr import apply as atom_asr_apply, fit as atom_asr_fit
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.tests import _setup, asr_clean, hann_at, iclabel_clean, kept, subjects

PRESETS = set()
NOISE = ('eye', 'muscle', 'electrode', 'line')
TITLE = {'eye': 'eye (blink + step)\nfrontal 1-10 Hz power', 'muscle': 'muscle burst\n30-90 Hz power', 'electrode': 'electrode pop\npeak amplitude, squared', 'line': 'line 60 Hz\n59-61 Hz power'}
FPZ, T7, T8 = (0.00011, 0.08825, -0.00171), (-0.08416, -0.01602, -0.00935), (0.08508, -0.01502, -0.00949)
EMG = butter(4, [30, 90], btype='band', fs=FS, output='sos')
BASE = ('backbone64', 'multi-task 0.5', 'multi-task 0.1')
OTHER = {'ASR': ('#1baf7a', '*'), 'ICLabel': ('#eda100', 'D'), 'backbone64': ('#555555', '^')}


def add_args(flags):
    flags.add('--syn-methods', nargs='+', help='task_filter_noise_removal_synthetic: only these methods (default all), e.g. "ASR atoms 1.5" "ASR atoms 1.5 + atoms -line"')
    flags.add('--syn-tag', default='', help='task_filter_noise_removal_synthetic: suffix of the output numbers file, to keep a partial run apart')
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--syn-dataset', default='PhysionetMI', help='task_filter_noise_removal_synthetic: the clean dataset the noise is added to (not one the filter was built from)')
    flags.add('--syn-subjects', type=int, default=8, help='task_filter_noise_removal_synthetic: subjects (spread out over the dataset)')
    flags.add('--syn-windows', type=int, default=24, help='task_filter_noise_removal_synthetic: windows per subject')
    flags.add('--syn-levels', type=float, nargs='+', default=[1, 3, 5], help='task_filter_noise_removal_synthetic: dB the scored quantity rises over the clean window')


def shapes(kind, B, C, T, xyz, rng):
    """Noise shapes [B, C, T] (scaled later)."""
    near = lambda p, tau: np.exp(-np.linalg.norm(xyz - np.array(p), axis=1) / tau)
    t = np.arange(T)
    if kind == 'eye':
        blink = np.stack([hann_at(T, rng.integers(T // 5, 4 * T // 5), 0.4 * FS) for _ in range(B)])
        step = np.stack([np.tanh((t - rng.integers(T // 5, 4 * T // 5)) / (0.05 * FS)) for _ in range(B)])
        lateral = np.sign(xyz[:, 0])[None, :, None] * near(FPZ, 0.1)[None, :, None]
        sign = rng.choice([-1, 1], B)[:, None, None]
        return near(FPZ, 0.06)[None, :, None] * blink[:, None] + 0.5 * sign * lateral * step[:, None]
    if kind == 'muscle':
        emg = sosfiltfilt(EMG, rng.standard_normal((B, C, T)), axis=-1)
        side = np.where(rng.random(B)[:, None] < 0.5, near(T7, 0.04)[None], near(T8, 0.04)[None]) + 0.15       # one temporal side, a diffuse part everywhere
        env = np.stack([hann_at(T, rng.integers(T // 4, 3 * T // 4), FS) for _ in range(B)])[:, None]
        return side[:, :, None] * emg * env
    if kind == 'electrode':
        out = np.zeros((B, C, T)); ch = rng.integers(0, C, B); t0 = rng.integers(T // 5, 4 * T // 5, B)
        for b in range(B):
            u = np.maximum(t - t0[b], 0) / (0.3 * FS)
            out[b, ch[b]] = rng.choice([-1, 1]) * np.where(t >= t0[b], np.exp(-u), 0.0)
        return out
    return (0.5 + rng.random((1, C, 1))) * np.sin(2 * np.pi * (60 * t / FS + rng.random((B, 1, 1))))


def quantity(kind, v, fp):
    """The scored quantity of every window of v [N, C, T] -> [N]."""
    f = lambda s, a: sosfiltfilt(s, a, axis=-1)
    if kind == 'eye':
        return (f(SLOW, v[:, fp].mean(1)) ** 2).mean(-1)
    if kind == 'muscle':
        return (f(HIGH, v) ** 2).mean((1, 2))
    if kind == 'line':
        return (f(LINE, v) ** 2).mean((1, 2))
    return np.percentile(np.abs(v).reshape(len(v), -1), 99.9, axis=1) ** 2


def scale(kind, x, n, db, fp):
    """n * a per window so that the scored quantity of x + a*n is `db` over x's own."""
    qx = quantity(kind, x, fp)
    if kind != 'electrode':
        return n * np.sqrt(qx * (10 ** (db / 10) - 1) / np.maximum(quantity(kind, n, fp), 1e-30))[:, None, None]
    lo, hi = np.zeros(len(x)), np.full(len(x), 100 * np.sqrt(qx.max()))
    for _ in range(25):                                                       # bisection: the percentile is not additive
        a = (lo + hi) / 2
        up = quantity(kind, x + a[:, None, None] * n, fp) > qx * 10 ** (db / 10)
        hi = np.where(up, a, hi); lo = np.where(up, lo, a)
    return n * ((lo + hi) / 2)[:, None, None]


def compute(ctx):
    dst = ctx.args.syn_dataset
    out, tf = filter_context(ctx, ctx.args.filter_source)
    if dst in tf.filt['sources']:
        raise SystemExit(f'{dst} is one of the datasets the filter was built from: pick another --syn-dataset')
    dst_r, dst_c, sel, encode, ops = _setup(tf, dst); dec = tf.backbone.decode
    lab = {m['label']: [m['cartesian'][k] for k in 'xyz'] for m in MONT}; xyz = np.array([lab[n] for n in dst_c])
    up = [n.upper() for n in dst_c]; fp = [up.index(n) for n in ('FP1', 'FP2')]
    atom_keep = {}
    if os.path.exists(ctx.numbers('atom_labels.npz')):
        zl = load_npz(ctx.numbers('atom_labels.npz')); top = [str(zl['classes'][i]) for i in zl['probs'].argmax(1)]
        atom_keep = {n: torch.tensor([t not in g for t in top], dtype=torch.float32) for n, g in ATOM_GROUPS.items() if n in ('atoms -eye', 'atoms -muscle', 'atoms -line', 'atoms -artifacts', 'atoms brain only')}
    shares = [q for q in ops if any(abs(q - a) < 1e-9 for a in (0.5, 0.1))]

    wanted = ctx.args.syn_methods; want = lambda m: not wanted or m in wanted
    asr_fit = {}

    def clean_all(xin, own=False):                                             # xin [N, C, T] -> {method: output}; own: calibrate the atom-space ASR on these (clean) windows
        T = xin.shape[-1]; g = encode(xin, True); sig = {'backbone64': dec(g[:, :, sel])[..., :T]}
        for q in shares:
            sig[f'multi-task {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', ops[q], g))[..., :T]
        for n, m in atom_keep.items():
            sig[n] = dec(g[:, :, sel] * m.to(g.device, g.dtype).view(1, 1, 1, -1, 1))[..., :T]
        if 'atoms -line' in atom_keep:                                         # ASR in the qlet space (task_filter_atom_asr), calibrated on the clean copies
            if own:
                asr_fit['m'] = atom_asr_fit(g[:, :, sel])
            ga, _ = atom_asr_apply(g[:, :, sel], asr_fit['m'], 1.5)
            sig['ASR atoms 1.5'] = dec(ga)[..., :T]
            sig['ASR atoms 1.5 + atoms -line'] = dec(ga * atom_keep['atoms -line'].view(1, 1, 1, -1, 1))[..., :T]
        return {m: w for m, w in sig.items() if want(m)}

    files = subjects(dst); order = np.argsort((np.arange(len(files)) * 0.6180339887) % 1)[:ctx.args.syn_subjects]
    rng = np.random.default_rng(0); res = {}
    for f in (files[i] for i in order):
        z = np.load(f); T = int(z['valid_end'][0]); data = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
        x = data[rng.permutation(len(data))[:ctx.args.syn_windows]]; x = x / x.std((1, 2), keepdims=True)
        calib = x.astype(np.float32) * 1e-5
        clean = clean_all(x, own=True)
        if want('ASR'):
            clean['ASR'] = asr_clean(calib, calib, dst_c).astype(np.float64) * 1e5
        if want('ICLabel'):
            clean['ICLabel'] = iclabel_clean(calib, dst_c)[0].astype(np.float64) * 1e5
        print(f'  {dst} subject {os.path.basename(f).split("_")[0]}: {len(x)} windows', flush=True)
        for kind in NOISE:
            n0 = shapes(kind, len(x), len(dst_c), T, xyz, rng)
            for db in ctx.args.syn_levels:
                n = scale(kind, x, n0, db, fp); w1 = x + n
                sig = clean_all(w1)
                if want('ASR'):
                    sig['ASR'] = asr_clean(w1.astype(np.float32) * 1e-5, calib, dst_c).astype(np.float64) * 1e5
                if want('ICLabel'):
                    sig['ICLabel'] = iclabel_clean(w1.astype(np.float32) * 1e-5, dst_c)[0].astype(np.float64) * 1e5
                qin, qx = quantity(kind, w1, fp).mean(), quantity(kind, x, fp).mean()
                for m, w in sig.items():
                    s_, a_ = kept(w, x, n); sc_, _ = kept(clean[m], x, np.zeros_like(x))
                    row = res.setdefault((kind, db, m), {})
                    for k, v in (('artifact', quantity(kind, w, fp).mean() / qin), ('control', quantity(kind, clean[m], fp).mean() / qx),
                                 ('signal', s_), ('leak', a_), ('signal_clean', sc_)):
                        row.setdefault(k, []).append(float(v))
    methods = list(dict.fromkeys(m for (_, _, m) in res)); arrays = {}
    for kind in NOISE:
        for db in ctx.args.syn_levels:
            for k in ('artifact', 'control', 'signal', 'leak', 'signal_clean'):
                arrays[f'{kind}__{db:g}__{k}'] = np.array([res[(kind, db, m)][k] for m in methods])
    save_npz(ctx.numbers('filter', ctx.args.filter_source, f'synthetic_{dst}{ctx.args.syn_tag}.npz'), methods=np.array(methods), levels=np.array(ctx.args.syn_levels), **arrays)
    mid = ctx.args.syn_levels[len(ctx.args.syn_levels) // 2]
    d = lambda v: 10 * np.log10(max(float(np.mean(v)), 1e-4))
    for kind in NOISE:
        print(f'\n  {kind} at {mid:g} dB over clean: artifact power change dB (control) | signal kept | noise left')
        for m in methods:
            r = res[(kind, mid, m)]
            print(f'    {m:16s} {d(r["artifact"]):6.1f} ({d(r["control"]):5.1f}) | {np.mean(r["signal"]):5.2f} | {np.mean(r["leak"]):5.2f}')


def draw(ctx):
    if not ctx.redraw:                                                      # numbers only: the group panels draw the figures; a single backbone's on request (--redraw)
        return
    dst = ctx.args.syn_dataset
    path = ctx.numbers('filter', ctx.args.filter_source, f'synthetic_{dst}.npz')
    if not os.path.exists(path):
        return
    z = load_npz(path); methods = [str(m) for m in z['methods']]; levels = [float(v) for v in z['levels']]; mid = levels[len(levels) // 2]
    colors = {'backbone64': '#555555', 'multi-task 0.5': '#eb6834', 'multi-task 0.1': '#6d1d05', 'ASR': '#1baf7a', 'ICLabel': '#eda100', **{m: c for m, (c, _) in ATOM_COLORS.items()}}
    mark = {m: v[1] for m, v in {**OTHER, **ATOM_COLORS}.items()}
    get = lambda kind, db, k, m: z[f'{kind}__{db:g}__{k}'][methods.index(m)]
    mean = lambda a: float(np.nanmean(a)); se = lambda a: float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    dbm = lambda a: 10 * np.log10(max(mean(a), 1e-4))
    fig, ax = plt.subplots(3, len(NOISE), figsize=(4.2 * len(NOISE), 12.5))
    for j, kind in enumerate(NOISE):
        a = ax[0, j]
        for m in methods:
            a.scatter([dbm(get(kind, mid, 'artifact', m))], [mean(get(kind, mid, 'signal', m))], color=colors[m], marker=mark.get(m, 'o'), s=70, zorder=3, label=m)
        a.scatter([0], [1], color='#555555', marker='s', s=60, label='input')
        a.set_xlabel('noise power change (dB)'); a.set_ylabel('signal kept (1 = all, exact)'); a.set_title(f'{TITLE[kind]}\n{mid:g} dB over the clean window', fontsize=9, fontweight='bold')
        a.axhline(1, color='k', lw=.4, ls=':'); a.axvline(0, color='k', lw=.4, ls=':')
        b = ax[1, j]; x = np.arange(len(methods))
        b.bar(x - .2, [dbm(get(kind, mid, 'artifact', m)) for m in methods], .38, color=[colors[m] for m in methods], yerr=[se(10 * np.log10(np.maximum(get(kind, mid, 'artifact', m), 1e-4))) for m in methods], capsize=2)
        b.bar(x + .2, [dbm(get(kind, mid, 'control', m)) for m in methods], .38, color='#cccccc')
        b.axhline(0, color='k', lw=.6); b.set_xticks(x); b.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
        b.set_ylabel('power change (dB; 0 = unchanged)'); b.set_title('coloured: noisy windows, grey: control (clean windows)', fontsize=7)
        c = ax[2, j]
        for m in methods:
            c.errorbar(levels, [20 * np.log10(max(mean(get(kind, db, 'signal', m)), 1e-2) / max(mean(get(kind, db, 'leak', m)), 1e-2)) for db in levels], color=colors[m], marker=mark.get(m, 'o'), ms=5, lw=1.2, label=m)
        c.set_xlabel('noise strength (dB over the clean window)'); c.axhline(0, color='k', lw=.6); c.set_ylabel('SNR change (dB; > 0 = cleaner)'); c.set_xticks(levels)
    ax[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle(textwrap.fill(f'Artificial noise on {dst} at TUAR strength (1 / 3 / 5 dB over clean): {ctx.args.filter_source} task filter and atom removal against ASR and ICLabel. Row 1: signal kept against noise power change, as in artifacts_TUAR. Row 2: power change with its control. Row 3: signal-to-noise change, 20 log10(signal kept / noise left), by noise strength.', width=int(24 * len(NOISE))), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ctx.figure_in('filter', ctx.args.filter_source, 'noise_removal_synthetic.png'); fig.savefig(out, dpi=110); plt.close(fig)
    print(f'  -> {out}')
