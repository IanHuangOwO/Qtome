"""task_filter_blinks: the task filter (--filter-source) on REAL eye blinks, against ASR and ICLabel -> pretrain/analysis/filter/<source>/blinks_<target>.npz,
analysis/filter/<source>/blinks_<target>.png. Builds the filter first if it is not saved.

--target (default SEED_V, a finetune dataset with Fp1 and Fp2) is read as recorded: no artifact is added. Blink trials are found from the
frontal channels (1-10 Hz band of the Fp1/Fp2 mean peaks above --blink-uv microvolts and at least 3 times the median peak of the other channels), clean
trials have a peak below --clean-uv. Signals per subject: input, the 64-channel reconstruction without filter (backbone64), the multi-task filter at each
share, ASR (calibrated on the subject's clean trials) and ICLabel (ICA on the selected trials). Scores, mean +- SE over --subjects subjects:
  blink left   RMS of the 1-10 Hz frontal signal on blink trials after / before (lower removes more of the blink);
  brain kept   correlation of the output with the input on clean trials (higher damages less where nothing needs removing);
  brain power  8-30 Hz power of the non-frontal channels on blink trials after / before.
Figure: a blink trial in Fp1 and Cz, the blink-locked average at Fp1 over all blink trials, and blink left against brain kept."""
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from panels.core.numbers import load_npz, save_npz
from panels.shared.task_filter.cells import FS
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.tests import _setup, asr_clean, iclabel_clean, subjects

PRESETS = set()
LP = butter(4, [1, 10], btype='band', fs=FS, output='sos')
ALPHA_BETA = butter(4, [8, 30], btype='band', fs=FS, output='sos')
HALF = int(0.6 * FS)                                                      # blink-locked average: +-0.6 s


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--target', nargs='+', help='dataset with Fp1 and Fp2 (default SEED_V; its channels must all be in the 10-10 montage)')
    flags.add('--blink-subjects', type=int, default=4, help='task_filter_blinks: subjects (evenly spread over the dataset)')
    flags.add('--blink-trials', type=int, default=30, help='task_filter_blinks: blink trials and clean trials per subject')
    flags.add('--blink-uv', type=float, default=100.0, help='task_filter_blinks: a blink trial peaks above this (uV, 1-10 Hz frontal mean)')
    flags.add('--clean-uv', type=float, default=40.0, help='task_filter_blinks: a clean trial peaks below this')


def find_trials(x_uv, fp, rest, blink_uv, clean_uv, n):
    """Trials [N, C, T] in microvolts -> (blink indices, clean indices, peak sample of each trial's frontal 1-10 Hz mean)."""
    front = sosfiltfilt(LP, x_uv[:, fp].mean(1), axis=-1)
    peak = np.abs(front).max(1)
    other = np.median(np.abs(sosfiltfilt(LP, x_uv[:, rest], axis=-1)).max(-1), axis=1)
    blink = np.flatnonzero((peak >= blink_uv) & (peak >= 3 * other))
    clean = np.flatnonzero(peak < clean_uv)
    return blink[np.argsort(-peak[blink])][:n], clean[:n], np.abs(front).argmax(1)


def rms(a):
    return np.sqrt((a ** 2).mean(axis=(-2, -1)))


def compute(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    for dst in ctx.args.target or ['SEED_V']:
        dst_r, dst_c, sel, encode, ops = _setup(tf, dst); dec = tf.backbone.decode
        up = [n.upper() for n in dst_c]
        fp = [up.index(n) for n in ('FP1', 'FP2') if n in up]
        if len(fp) < 2:
            raise ValueError(f'{dst} has no Fp1 and Fp2 among {dst_c}')
        rest = [i for i, n in enumerate(up) if n not in ('FP1', 'FP2', 'FPZ', 'AF7', 'AF8', 'AF3', 'AF4', 'FZ')]
        show = [dst_c[up.index(n)] for n in ('FP1', 'CZ') if n in up]
        files = subjects(dst); pick = np.linspace(0, len(files) - 1, min(ctx.args.blink_subjects, len(files))).round().astype(int)
        methods, per, avg, count, example = None, {}, None, 0, None
        for si, f in enumerate(files[i] for i in pick):
            z = np.load(f); T = int(z['valid_end'][0])
            raw = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
            blink, clean, peak_at = find_trials(raw * 1e6, fp, rest, ctx.args.blink_uv, ctx.args.clean_uv, ctx.args.blink_trials)
            print(f'  {dst} subject {os.path.basename(f).split("_")[0]}: {len(blink)} blink trials, {len(clean)} clean trials')
            if len(blink) < 5 or len(clean) < 8:
                continue
            idx = np.concatenate([blink, clean]); nb = len(blink)
            sd = raw[idx].std((1, 2), keepdims=True); x = raw[idx] / sd
            g = encode(x, True); sig = {'input': x, 'backbone64': dec(g[:, :, sel])[..., :T]}
            for q, op in ops.items():
                sig[f'multi-task {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', op, g))[..., :T]
            sig['ASR'] = asr_clean(x.astype(np.float32) * 1e-5, x[nb:].astype(np.float32) * 1e-5, dst_c).astype(np.float64) * 1e5
            sig['ICLabel'] = iclabel_clean(x.astype(np.float32) * 1e-5, dst_c)[0].astype(np.float64) * 1e5
            methods = list(sig)
            front_in = sosfiltfilt(LP, sig['input'][:nb][:, fp].mean(1), axis=-1)
            sign = np.sign(front_in[np.arange(nb), peak_at[blink]])
            if avg is None:
                avg = {m: np.zeros(2 * HALF) for m in methods}
            for m, w in sig.items():
                front = sosfiltfilt(LP, w[:nb][:, fp].mean(1), axis=-1)
                bp = lambda a: sosfiltfilt(ALPHA_BETA, a, axis=-1)
                per.setdefault(m, {}).setdefault('blink left', []).append(float(rms(front) .mean() / rms(front_in).mean()))
                per[m].setdefault('brain kept', []).append(float(np.mean([np.corrcoef(w[nb + k].ravel(), x[nb + k].ravel())[0, 1] for k in range(len(clean))])))
                per[m].setdefault('brain power', []).append(float((bp(w[:nb][:, rest]) ** 2).mean() / (bp(x[:nb][:, rest]) ** 2).mean()))
                for k in range(nb):                                       # blink-locked average at the first frontal channel, uV, blink peak up
                    a = int(np.clip(peak_at[blink[k]], HALF, T - HALF))
                    avg[m] += sign[k] * w[k, fp[0], a - HALF:a + HALF] * sd[k, 0, 0] * 1e6
            count += nb
            if example is None or si == 0:                                # the median-peak blink trial of the first subject used
                k = nb // 2
                ch = [dst_c.index(c) for c in show]
                example = {m: w[k][ch] * sd[k, 0, 0] * 1e6 for m, w in sig.items()}
        if not per:
            raise RuntimeError(f'no subject of {dst} has enough blink and clean trials: lower --blink-uv or raise --blink-subjects')
        save_npz(ctx.numbers('filter', ctx.args.filter_source, f'blinks_{dst}.npz'), methods=np.array(methods), show=np.array(show),
                 blinks=np.array(count), t=(np.arange(2 * HALF) - HALF) / FS, **{k: np.array([per[m][k] for m in methods]) for k in ('blink left', 'brain kept', 'brain power')},
                 avg=np.stack([avg[m] / count for m in methods]), example=np.stack([example[m] for m in methods]))
        n = len(next(iter(per.values()))['blink left'])
        print(f'\n{dst}: {n} subjects, {count} blink trials, mean +- SE over subjects')
        print(f'  {"":18s}' + ''.join(f'{k:>16s}' for k in ('blink left', 'brain kept', 'brain power')))
        for m in methods:
            print(f'  {m:18s}' + ''.join(f'{np.mean(per[m][k]):8.3f} +-{np.std(per[m][k], ddof=1) / np.sqrt(n) if n > 1 else 0:5.3f}' for k in ('blink left', 'brain kept', 'brain power')))


def draw(ctx):
    for dst in ctx.args.target or ['SEED_V']:
        path = ctx.numbers('filter', ctx.args.filter_source, f'blinks_{dst}.npz')
        if not os.path.exists(path):
            continue
        z = load_npz(path)
        methods = [str(m) for m in z['methods']]
        colors = {'input': '#000000', 'backbone64': '#888888', 'ASR': '#1baf7a', 'ICLabel': '#eda100'}
        shares = [m for m in methods if m.startswith('multi-task')]
        for k, m in enumerate(sorted(shares, key=lambda s: -float(s.split()[-1]))):
            colors[m] = ('#eb6834', '#d04a1a', '#9c2f0a', '#6d1d05')[k % 4]
        fig = plt.figure(figsize=(16, 5.6)); gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
        t = np.arange(z['example'].shape[-1]) / FS
        for r, ch in enumerate([str(c) for c in z['show']]):
            ax = fig.add_subplot(gs[r, 0])
            for i, m in enumerate(methods):
                if m in ('input', 'ASR', 'ICLabel', min(shares, key=lambda s: float(s.split()[-1]), default=None)):
                    ax.plot(t, z['example'][i, r], color=colors[m], lw=1.2 if m == 'input' else 0.9, label=m)
            ax.set_ylabel(f'{ch} (uV)'); ax.set_title('one blink trial' if r == 0 else '', fontsize=10, fontweight='bold')
            if r == 0:
                ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel('s')
        ax = fig.add_subplot(gs[:, 1])
        for i, m in enumerate(methods):
            ax.plot(z['t'], z['avg'][i], color=colors.get(m, '#888888'), lw=1.6 if m == 'input' else 1.0, label=m)
        ax.set_xlabel('time from the blink peak (s)'); ax.set_ylabel(f'{z["show"][0]} (uV, blink peak up)')
        ax.set_title(f'blink-locked average, {int(z["blinks"])} blinks', fontsize=10, fontweight='bold'); ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[:, 2])
        for i, m in enumerate(methods):
            xs, ys = z['blink left'][i], z['brain kept'][i]
            n = len(xs); se = lambda v: np.std(v, ddof=1) / np.sqrt(n) if n > 1 else 0
            ax.errorbar(np.mean(xs), np.mean(ys), xerr=se(xs), yerr=se(ys), fmt='o', color=colors.get(m, '#888888'), capsize=2)
            ax.annotate(m, (np.mean(xs), np.mean(ys)), fontsize=7, xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel('blink left on blink trials (input = 1; lower removes more)'); ax.set_ylabel('brain kept on clean trials (correlation with input)')
        ax.set_title('removal against damage (top left is best)', fontsize=10, fontweight='bold')
        fig.suptitle(f'Real blinks in {dst}: {ctx.args.filter_source} task filter against ASR and ICLabel', fontweight='bold')
        fig.tight_layout()
        out = ctx.figure_in('filter', ctx.args.filter_source, f'blinks_{dst}.png')
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f'  -> {out}')
