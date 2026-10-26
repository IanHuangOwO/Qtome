"""task_filter_artifacts: the task filter (--filter-source) on the artifacts annotated in TUAR (eye movement `eyem`, muscle `musc`, electrode `elec`,
and the mixed labels `eyem_musc` and `musc_elec`) and on TUAR's real 60 Hz line noise, against ASR and ICLabel ->
pretrain/analysis/filter/<source>/artifacts_TUAR.npz, analysis/filter/<source>/artifacts_TUAR.png. Builds the filter first if it is not saved.
Nothing is added to the recordings.

Windows are TUAR's own 5 s windows (datas/finetune/TUAR, annotation files in the loader's layout, checked against the cache's artifact label
for every patient; a patient that does not match is left out). A window belongs to a label when that exact label covers at least half of it and
any other label under 10%; clean = no annotation at all. Per patient, the signals: input, the 64-channel reconstruction without filter
(backbone64), the multi-task filter at each share (--art-shares), ASR (calibrated on the patient's clean windows) and ICLabel (ICA on the
selected windows). Each figure column is one score on one window set, as output / input POWER (1 = unchanged, 0.1 = -10 dB), mean +- SE over the
patients that have enough windows of the label; a mixed label has one column per score it carries:
  eye left      power of the 1-10 Hz Fp1/Fp2 mean                        on eyem and eyem_musc windows   control: the same on clean windows
  muscle left   30-90 Hz power                                           on musc, eyem_musc and musc_elec windows   control: the same on clean windows
  peak left     squared 99.9th percentile of |signal|                    on elec and musc_elec windows    control: the same on clean windows
  line left     59-61 Hz power on clean windows                          control: 8-30 Hz power on clean windows (the brain band)
  brain kept    correlation of the output with the input on clean windows (not a ratio)
A method that only smooths or low-passes lowers the control as much as the artifact score; removal specific to the artifact lowers
the artifact score and leaves the control. Figure rows: removal against damage; artifact score against its control in dB; the same in linear ratio."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, welch

from panels.core.numbers import load_npz, save_npz
from IO.channels import get_standard_coords
from panels.shared.atom_dictionary import atom_templates, peak_freq
from panels.shared.task_filter import artifact_views as views
from panels.shared.task_filter.cells import FS, cell_dir
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.fit import SOURCES
from panels.shared.task_filter.tests import _setup, asr_clean, iclabel_clean, subjects

PRESETS = {'standard'}
SLOW = butter(4, [1, 10], btype='band', fs=FS, output='sos')
HIGH = butter(4, [30, 90], btype='band', fs=FS, output='sos')
LINE = butter(4, [59, 61], btype='band', fs=FS, output='sos')
BRAIN = butter(4, [8, 30], btype='band', fs=FS, output='sos')
KINDS = ('eyem', 'musc', 'elec', 'eyem_musc', 'musc_elec')                 # the TUAR labels with a window set
SETS = (*KINDS, 'clean')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 70), (70, 100))   # the decodability features: log band power per channel
COLUMNS = (('eyem', 'eye left'), ('musc', 'muscle left'), ('elec', 'peak left'), ('eyem_musc', 'eye left'), ('eyem_musc', 'muscle left'),
           ('musc_elec', 'muscle left'), ('musc_elec', 'peak left'), (None, 'line left'))     # (window set, score) per figure column
SCORE_KIND = {'eye left': 'eyem', 'muscle left': 'musc', 'peak left': 'elec', 'line left': None}     # which measure a score is


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--art-patients', type=int, default=8, help='task_filter_artifacts: TUAR patients wanted per artifact type (patients tried in a spread-out order)')
    flags.add('--art-shares', type=float, nargs='+', default=[0.5, 0.1], help='task_filter_artifacts: the multi-task filter shares tested')
    flags.add('--art-windows', type=int, default=20, help='task_filter_artifacts: windows per artifact type and patient (clean: 1.5 x this)')


def tuar_coverage(sub):
    """-> ({label: share of each window under an annotated event of that exact label}, share under any annotated event), in the loader's window layout."""
    import mne
    mne.set_log_level('ERROR')
    base = 'datas/finetune/TUAR/'
    cov, anyc = {k: [] for k in KINDS}, []
    for r in json.load(open(base + 'metadata.json'))['data_structure'][str(sub)]['runs']:
        try:
            raw = mne.io.read_raw_edf(base + r['file'], preload=False, verbose=False)
        except Exception:
            continue                                                          # the loader skips unreadable runs the same way
        n, w = int(raw.n_times * FS / raw.info['sfreq']) // (5 * FS), 5 * FS
        m = {k: np.zeros(n * w, bool) for k in (*KINDS, 'any')}
        for line in open(base + r['ann']):
            p = line.strip().split(',')
            if len(p) >= 4 and p[1][:1].isdigit() and p[3].lower() != 'null':
                a, b = max(0, int(float(p[1]) * FS)), min(n * w, int(float(p[2]) * FS))
                m['any'][a:b] = True
                if p[3].lower() in KINDS:
                    m[p[3].lower()][a:b] = True
        for i in range(n):
            for k in KINDS:
                cov[k].append(m[k][i * w:(i + 1) * w].mean())
            anyc.append(m['any'][i * w:(i + 1) * w].mean())
    return {k: np.array(v) for k, v in cov.items()}, np.array(anyc)


def power(a):
    return (a ** 2).mean()


def ratios(w, x, idx, kind, fp):
    """The artifact score (and its control) of one signal w on the windows idx; x is the input of the same windows."""
    f = lambda s, v: sosfiltfilt(s, v, axis=-1)
    if kind == 'eyem':
        a = lambda v: power(f(SLOW, v[idx][:, fp].mean(1)))
        return a(w) / a(x)
    if kind == 'musc':
        return power(f(HIGH, w[idx])) / power(f(HIGH, x[idx]))
    q = lambda v: np.mean(np.percentile(np.abs(v[idx]).reshape(len(idx), -1), 99.9, axis=1))
    return (q(w) / q(x)) ** 2                                              # peak amplitude squared: a power-like ratio, comparable with the others


def compute(ctx):
    bb = os.path.dirname(os.path.dirname(os.path.dirname(ctx.checkpoint)))
    missing = [d for d in (ctx.args.filter_sources or SOURCES) if not os.path.exists(f'{cell_dir(bb, d)}/artifacts/group_eval.json')]
    if missing and not os.path.exists(f'{ctx.numbers_dir}/filter/{ctx.args.filter_source}/filter.pt'):
        print(f'  skipped: no loso impute64 cell for {missing} (the filter needs them; see task_filter_build)'); return
    out, tf = filter_context(ctx, ctx.args.filter_source)
    dst = 'TUAR'
    dst_r, dst_c, sel, encode, ops = _setup(tf, dst); dec = tf.backbone.decode
    up = [n.upper() for n in dst_c]; fp = [up.index(n) for n in ('FP1', 'FP2')]
    files = subjects(dst); order = np.argsort((np.arange(len(files)) * 0.6180339887) % 1)[:150]
    methods, per, cnt = None, {}, {k: 0 for k in (*KINDS, 'clean')}
    nwin = {k: 0 for k in KINDS}
    ex = {'feat': {}, 'psd': {}, 'chan': {}, 'q': {}, 'atom_out': {}, 'atom_in': [], 'set': [], 'patient': [], 'example': {}}
    san = lambda m: m.replace(' ', '_')
    ex_ch = [dst_c.index(n) for n in ('Fp1', 'Cz', 'T7', 'T3', 'C3') if n in dst_c][:3]
    for f in (files[i] for i in order):
        if all(cnt[k] >= ctx.args.art_patients for k in KINDS):
            break
        z = np.load(f); T = int(z['valid_end'][0]); sub = int(os.path.basename(f).split('_')[0])
        raw = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
        cov, anyc = tuar_coverage(sub)
        if len(anyc) != len(raw) or not np.array_equal((anyc >= 0.5).astype(int), z['labels']):
            print(f'  TUAR patient {sub}: annotation layout does not match the cache: left out'); continue
        rng = np.random.default_rng(sub); nw = ctx.args.art_windows
        idx = {k: rng.permutation(np.flatnonzero((cov[k] >= 0.5) & (anyc - cov[k] < 0.1)))[:nw] for k in KINDS}
        idx['clean'] = rng.permutation(np.flatnonzero(anyc == 0))[:int(1.5 * nw)]
        want = [k for k in KINDS if len(idx[k]) >= 5 and cnt[k] < ctx.args.art_patients]
        if len(idx['clean']) < 8 or not want:
            continue
        pos, allidx = {}, []
        for k in (*want, 'clean'):
            pos[k] = np.arange(len(allidx), len(allidx) + len(idx[k])); allidx += list(idx[k])
        allidx = np.array(allidx)
        sd = raw[allidx].std((1, 2), keepdims=True); x = raw[allidx] / sd
        print(f'  {dst} patient {sub}: ' + ', '.join(f'{k} {len(pos[k])}' for k in pos))
        g = encode(x, True); sig = {'input': x, 'backbone64': dec(g[:, :, sel])[..., :T]}
        for q, op in ops.items():
            if not any(abs(q - a) < 1e-9 for a in ctx.args.art_shares):
                continue
            sig[f'multi-task {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', op, g))[..., :T]
        calib = x[pos['clean']].astype(np.float32) * 1e-5
        sig['ASR'] = asr_clean(x.astype(np.float32) * 1e-5, calib, dst_c).astype(np.float64) * 1e5
        sig['ICLabel'] = iclabel_clean(x.astype(np.float32) * 1e-5, dst_c)[0].astype(np.float64) * 1e5
        methods = list(sig); cl = pos['clean']
        N = len(allidx)
        ex['set'].append(np.concatenate([np.full(len(pos[k]), SETS.index(k)) for k in pos])); ex['patient'].append(np.full(N, sub))
        gs = g[:, :, sel]
        ex['atom_in'].append(gs.pow(2).sum(-1).mean((1, 2)).float().cpu().numpy())               # [N, S] atom power per window
        for q, op in ops.items():
            if any(abs(q - a) < 1e-9 for a in ctx.args.art_shares):
                ex['atom_out'].setdefault(f'multi-task {q:g}', []).append(torch.einsum('scd,bndsx->bncsx', op, g).pow(2).sum(-1).mean((1, 2)).float().cpu().numpy())
        for k in pos:
            if k != 'clean' and k not in ex['example']:
                i0 = pos[k][len(pos[k]) // 2]
                ex['example'][k] = {m: (w[i0][ex_ch] * sd[i0, 0, 0] * 1e6).astype(np.float32) for m, w in sig.items()}
        for m, w in sig.items():
            fr, P = welch(w, fs=FS, nperseg=200, axis=-1)                           # [N, C, 101], 1 Hz bins
            bp = np.stack([P[..., (fr >= lo) & (fr < hi)].mean(-1) for lo, hi in BANDS], -1)
            ex['feat'].setdefault(m, []).append(np.log(bp + 1e-20).reshape(N, -1).astype(np.float32))
            ex['psd'].setdefault(m, []).append(P.mean(1).astype(np.float32)); ex['chan'].setdefault(m, []).append(P.sum(-1).astype(np.float32))
            ex['q'].setdefault(m, []).append(np.stack([(sosfiltfilt(SLOW, w[:, fp].mean(1), axis=-1) ** 2).mean(-1), (sosfiltfilt(HIGH, w, axis=-1) ** 2).mean((1, 2)),
                                                       np.percentile(np.abs(w).reshape(N, -1), 99.9, axis=1) ** 2], 1))
        for m, w in sig.items():
            d = per.setdefault(m, {})
            for k, score in COLUMNS:
                if k in want:
                    d.setdefault(f'{k}__{score}', []).append(float(ratios(w, x, pos[k], SCORE_KIND[score], fp)))
                    d.setdefault(f'{k}__{score}__control', []).append(float(ratios(w, x, cl, SCORE_KIND[score], fp)))
            lw = sosfiltfilt(LINE, w[cl], axis=-1); li = sosfiltfilt(LINE, x[cl], axis=-1)
            brain = lambda v: power(sosfiltfilt(BRAIN, v[cl], axis=-1))
            d.setdefault('line__line left', []).append(float(power(lw) / power(li)))
            d.setdefault('line__line left__control', []).append(float(brain(w) / brain(x)))
            d.setdefault('brain kept', []).append(float(np.mean([np.corrcoef(w[i].ravel(), x[i].ravel())[0, 1] for i in cl])))
        for k in want:
            cnt[k] += 1; nwin[k] += len(pos[k])
        cnt['clean'] += 1
    if not per:
        raise RuntimeError('no TUAR patient qualified: lower --art-windows or raise --art-patients')
    keys = list(next(iter(per.values())))
    arrays = {}
    for k in keys:                                                            # one row per method; patients vary by label, NaN pads
        n = max(len(per[m].get(k, [])) for m in methods)
        arrays[k.replace(' ', '_')] = np.array([per[m].get(k, []) + [np.nan] * (n - len(per[m].get(k, []))) for m in methods])
    cat = lambda v: np.concatenate(v)
    extra = {'win_set': cat(ex['set']), 'win_patient': cat(ex['patient']), 'atom_in': cat(ex['atom_in']), 'channels': np.array(dst_c),
             'coords': np.stack([get_standard_coords(n) for n in dst_c]), 'atom_peak': peak_freq(atom_templates(tf.backbone.model)[0]), 'ex_channels': np.array([dst_c[i] for i in ex_ch])}
    for m in methods:
        extra[f'feat__{san(m)}'] = cat(ex['feat'][m]); extra[f'psd__{san(m)}'] = cat(ex['psd'][m]); extra[f'chan__{san(m)}'] = cat(ex['chan'][m]); extra[f'q__{san(m)}'] = cat(ex['q'][m])
        for k, tr in ex['example'].items():
            extra[f'example__{k}__{san(m)}'] = tr[m]
    for m, v in ex['atom_out'].items():
        extra[f'atom_out__{san(m)}'] = cat(v)
    save_npz(ctx.numbers('filter', ctx.args.filter_source, f'artifacts_{dst}.npz'), methods=np.array(methods), keys=np.array(keys), **arrays, **extra)
    print(f'\n{dst}: patients per label ' + ', '.join(f'{k} {cnt[k]} ({nwin[k]} windows)' for k in KINDS) + f', clean {cnt["clean"]}' + '; mean +- SE over patients (artifact score | control)')
    se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    for k, score in (*COLUMNS[:-1], ('line', 'line left')):
        key = f'{k}__{score}'
        if key not in per[methods[0]]:
            continue
        print(f'\n  {score} on {k if k != "line" else "clean windows (real 60 Hz)"}')
        for m in methods:
            a, c = np.array(per[m][key]), np.array(per[m][key + '__control'])
            print(f'    {m:18s} {np.mean(a):6.3f} +-{se(a):5.3f} | {np.mean(c):6.3f} +-{se(c):5.3f}')
    print('\n  brain kept: ' + ', '.join(f'{m} {np.mean(per[m]["brain kept"]):.3f}' for m in methods))


DB_FLOOR = -40.0


def db(r):
    return np.maximum(10 * np.log10(np.maximum(r, 1e-12)), DB_FLOOR)


def draw(ctx):
    path = ctx.numbers('filter', ctx.args.filter_source, 'artifacts_TUAR.npz')
    if not os.path.exists(path):
        return
    z = load_npz(path)
    methods = [str(m) for m in z['methods']]
    keep = lambda m: not m.startswith('multi-task') or any(abs(float(m.split()[-1]) - a) < 1e-9 for a in ctx.args.art_shares)
    shown = [m for m in methods if m != 'input' and keep(m)]
    shares = sorted([m for m in shown if m.startswith('multi-task')], key=lambda s: -float(s.split()[-1]))
    col = lambda key, m: z[key.replace(' ', '_')][methods.index(m)]
    mean = lambda key, m: float(np.nanmean(col(key, m)))
    se = lambda key, m: float(np.nanstd(col(key, m), ddof=1) / np.sqrt(np.sum(~np.isnan(col(key, m)))))
    mean_db = lambda key, m: float(db(mean(key, m)))
    err = lambda key, m: [mean_db(key, m) - db(mean(key, m) - se(key, m)), db(mean(key, m) + se(key, m)) - mean_db(key, m)]
    points = {'backbone64': ('#555555', '^'), 'ASR': ('#1baf7a', '*'), 'ICLabel': ('#eda100', 'D')}
    what = {'eye left': 'frontal 1-10 Hz power', 'muscle left': '30-90 Hz power', 'peak left': 'peak amplitude, squared', 'line left': '59-61 Hz power'}
    name = {'eyem': 'eye movement (eyem)', 'musc': 'muscle (musc)', 'elec': 'electrode (elec)', 'eyem_musc': 'mixed eyem_musc', 'musc_elec': 'mixed musc_elec', None: 'line noise (real)'}
    cols = [(k if k else 'line', sc) for k, sc in COLUMNS]
    fig, ax = plt.subplots(3, len(cols), figsize=(3.7 * len(cols), 13.2))
    for j, (k, s) in enumerate(cols):
        key = f'{k}__{s}'; n = int(np.sum(~np.isnan(col(key, 'input'))))
        a = ax[0, j]
        xs, ys = [mean_db(key, m) for m in shares], [mean('brain kept', m) for m in shares]
        a.plot(xs, ys, 'o-', color='#eb6834', lw=2, ms=8, mec='white', mew=1.5, label='multi-task')
        for m, x_, y_ in zip(shares, xs, ys):
            a.annotate(m.split()[-1], (x_, y_), fontsize=9, xytext=(5, 4), textcoords='offset points')
        a.scatter([0], [1], color='#555555', marker='s', s=60, zorder=3, label='input')
        for m, (c, mk) in points.items():
            a.scatter([mean_db(key, m)], [mean('brain kept', m)], color=c, marker=mk, s=110 if mk == '*' else 60, zorder=3, label=m)
        a.axvline(0, color='k', lw=.5, ls=':'); a.set_xlabel('artifact power change (dB)'); a.set_ylabel('brain kept on clean windows (correlation)')
        a.set_title(f'{name[None if k == "line" else k]}\n{what[s]}\n{n} patients', fontsize=9, fontweight='bold')
        if j == 0:
            a.legend(fontsize=8, loc='lower left')
        b, c_ = ax[1, j], ax[2, j]; x = np.arange(len(shown))
        colors = ['#eb6834' if m.startswith('multi') else points[m][0] for m in shown]
        ctl = key + '__control'
        b.bar(x - .2, [mean_db(key, m) for m in shown], .38, yerr=np.array([err(key, m) for m in shown]).T, color=colors, capsize=2)
        b.bar(x + .2, [mean_db(ctl, m) for m in shown], .38, yerr=np.array([err(ctl, m) for m in shown]).T, color='#cccccc', capsize=2)
        lo = min(min(mean_db(key, m), mean_db(ctl, m)) for m in shown); hi = max(max(mean_db(key, m), mean_db(ctl, m)) for m in shown); pad = max(1.0, 0.1 * (hi - lo))
        b.axhline(0, color='k', lw=.6); b.set_ylim(max(DB_FLOOR - 1, lo - pad - 1), max(hi, 0) + pad + 1)
        b.set_title('coloured: artifact windows\ngrey: control, ' + ('8-30 Hz brain power' if k == 'line' else 'clean windows'), fontsize=7)
        b.set_ylabel('power change (dB; 0 = unchanged, negative = removed)')
        c_.bar(x - .2, [mean(key, m) for m in shown], .38, yerr=[se(key, m) for m in shown], color=colors, capsize=2)
        c_.bar(x + .2, [mean(ctl, m) for m in shown], .38, yerr=[se(ctl, m) for m in shown], color='#cccccc', capsize=2)
        c_.axhline(1, color='k', lw=.6); c_.set_ylabel('output power / input power (1 = unchanged)'); c_.set_title('same as the row above, linear scale', fontsize=7)
        for t in (b, c_):
            t.set_xticks(x); t.set_xticklabels(shown, rotation=40, ha='right', fontsize=7)
    fig.suptitle(f'TUAR artifacts: {ctx.args.filter_source} task filter ({", ".join(f"{a:g}" for a in ctx.args.art_shares)}) against ASR and ICLabel. Every score is output power / input power. '
                 f'Rows: removal against damage; artifact score against its control in dB (floor {DB_FLOOR:g} dB); the same in linear ratio. A mixed label has a column per score.', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = ctx.figure_in('filter', ctx.args.filter_source, 'artifacts_TUAR.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'  -> {out}')
    if 'win_set' not in z:
        print('  (numbers from an older run: the extra views need a fresh run)')
        return
    colors = {'input': '#000000', 'backbone64': '#555555', 'ASR': '#1baf7a', 'ICLabel': '#eda100'}
    for m, c in zip(shares, ('#eb6834', '#6d1d05', '#9c2f0a')):
        colors[m] = c
    fig_in = lambda name: ctx.figure_in('filter', ctx.args.filter_source, name)
    print(f'  -> {views.spectrum(z, shown, colors, fig_in("artifacts_TUAR_spectrum.png"))}')
    print(f'  -> {views.dose(z, shown, colors, fig_in("artifacts_TUAR_dose.png"))}')
    print(f'  -> {views.atoms(z, shares, fig_in("artifacts_TUAR_atoms.png"))}')
    path, res = views.decode(z, shown, colors, fig_in('artifacts_TUAR_decode.png'))
    print(f'  -> {path}')
    for k in views.LABELS:
        if (k, 'input') in res:
            print(f'     AUC {k:10s} ' + ', '.join(f'{m} {res[(k, m)]:.2f}' for m in ('input', *shown)))
    print(f'  -> {views.maps(z, shares, fig_in("artifacts_TUAR_maps.png"))}')
    print(f'  -> {views.examples(z, [m for m in shown if m in (*shares, "ASR", "ICLabel")], colors, fig_in("artifacts_TUAR_examples.png"))}')
    keys = [(f'{k}__{sc}', f'{name[None if k == "line" else k]}: {what[sc]}') for k, sc in cols]
    print(f'  -> {views.patients(z, shown, colors, keys, fig_in("artifacts_TUAR_patients.png"))}')
