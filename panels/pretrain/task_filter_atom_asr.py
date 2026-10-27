"""task_filter_atom_asr: artifact subspace reconstruction (ASR) in the Q-atom space. ASR calibrates on clean data, finds the directions in channel space that
carry more variance than the clean data ever did, and removes them. Here it runs on the Q-atom gains: for every Q-atom separately, the [C] vectors of its
gains (a and b of every patch) of the patient's clean windows give the clean channel covariance, its eigenvectors the components and the clean rms of
each component. In a test window every component whose rms over the window's patches exceeds --asr-cutoff x its clean rms is set to zero for that atom
and window (hard removal; the original ASR reconstructs the component from the clean mixing instead); the changed gains decode through the unchanged
atom bank. The patient's clean windows both calibrate and serve as the control, as for the time-domain ASR it is compared with (cutoff 20, asrpy).
Scored like task_filter_artifacts on TUAR (artifact score and clean-window control in dB, brain kept) next to backbone64, atoms -eye, the multi-task
filter at 0.1 and the time-domain ASR of the same run. -> pretrain/analysis/filter/<source>/atom_asr_TUAR<tag>.npz (same keys as artifacts_TUAR.npz)
and the share of components removed per window set (removed_<cutoff>__<label>); figure only with --redraw."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import sosfiltfilt

from panels.core.numbers import load_npz, save_npz
from panels.pretrain.task_filter_artifacts import ATOM_GROUPS, BRAIN, COLUMNS, KINDS, LINE, SCORE_KIND, power, ratios
from panels.pretrain.task_filter_zsubspace import dst_windows, stack
from panels.shared.task_filter.atom_asr import atom_asr
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.tests import _setup, asr_clean, subjects

PRESETS = set()
DB_FLOOR = -40.0


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--art-patients', type=int, default=8, help='task_filter_artifacts: TUAR patients wanted per artifact type (patients tried in a spread-out order)')
    flags.add('--art-windows', type=int, default=20, help='task_filter_artifacts: windows per artifact type and patient (clean: 1.5 x this)')
    flags.add('--asr-tag', default='', help='task_filter_atom_asr: suffix of the output file, to keep variants apart')
    flags.add('--asr-combo', type=float, nargs='*', default=[1.5, 2], help='task_filter_atom_asr: cutoffs also combined with atoms -line and the multi-task filter (ASR first)')
    flags.add('--asr-cutoffs', type=float, nargs='+', default=[3, 5, 10, 20], help='task_filter_atom_asr: removal thresholds in clean standard deviations')


def compute(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    columns = [(k, sc) for k, sc in COLUMNS if k is None or k in KINDS]
    dst_r, dst_c, sel, encode, ops = _setup(tf, 'TUAR'); dec = tf.backbone.decode
    up = [n.upper() for n in dst_c]; fp = [up.index(n) for n in ('FP1', 'FP2')]
    zl = load_npz(ctx.numbers('atom_labels.npz')); top = [str(zl['classes'][i]) for i in zl['probs'].argmax(1)]
    atom_keep = {n: torch.tensor([t not in ATOM_GROUPS[n] for t in top], dtype=torch.float32) for n in ('atoms -eye', 'atoms -line')}
    files = subjects('TUAR'); order = [files[i] for i in np.argsort((np.arange(len(files)) * 0.6180339887) % 1)[:150]]
    nw = ctx.args.art_windows
    per, cnt, rem = {}, {k: 0 for k in (*KINDS, 'clean')}, {}
    for f in order:
        if all(cnt[k] >= ctx.args.art_patients for k in KINDS):
            break
        w = dst_windows(f, nw, dst_r, dst_c)
        if w is None:
            continue
        sub, raw, idx = w
        want = [k for k in KINDS if len(idx[k]) >= 5 and cnt[k] < ctx.args.art_patients]
        if len(idx['clean']) < 8 or not want:
            continue
        pos, x = stack(raw, idx, (*want, 'clean')); T = x.shape[-1]; cl = pos['clean']
        g64 = encode(x, True); g = g64[:, :, sel]
        sig = {'backbone64': dec(g)[..., :T]}
        for q in (0.5, 0.1):
            if q in ops:
                sig[f'multi-task {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', ops[q], g64))[..., :T]
        for n, m in atom_keep.items():
            sig[n] = dec(g * m.view(1, 1, 1, -1, 1))[..., :T]
        sig['ASR (time)'] = asr_clean(x.astype(np.float32) * 1e-5, x[cl].astype(np.float32) * 1e-5, dst_c).astype(np.float64) * 1e5
        for c in ctx.args.asr_cutoffs:
            gc, share = atom_asr(g, torch.as_tensor(cl), c)
            sig[f'ASR atoms {c:g}'] = dec(gc)[..., :T]
            if c in ctx.args.asr_combo:                                      # the line-noise remover after ASR (ASR is per atom, so the order of the atom mask does not matter)
                sig[f'ASR atoms {c:g} + atoms -line'] = dec(gc * atom_keep['atoms -line'].view(1, 1, 1, -1, 1))[..., :T]
                g64c = g64.clone(); g64c[:, :, sel] = gc
                for q in (0.5, 0.1):
                    if q in ops:
                        sig[f'ASR atoms {c:g} + multi-task {q:g}'] = dec(torch.einsum('scd,bndsx->bncsx', ops[q], g64c))[..., :T]
            for k in pos:
                rem.setdefault(f'removed_{c:g}__{k}', []).append(float(share[pos[k]].mean()))
        print(f'  patient {sub}: ' + ', '.join(f'{k} {len(pos[k])}' for k in pos))
        for m, wv in sig.items():
            d = per.setdefault(m, {})
            for k, score in columns:
                if k in want:
                    d.setdefault(f'{k}__{score}', []).append(float(ratios(wv, x, pos[k], SCORE_KIND[score], fp)))
                    d.setdefault(f'{k}__{score}__control', []).append(float(ratios(wv, x, cl, SCORE_KIND[score], fp)))
            d.setdefault('line__line left', []).append(float(power(sosfiltfilt(LINE, wv[cl], axis=-1)) / power(sosfiltfilt(LINE, x[cl], axis=-1))))
            d.setdefault('line__line left__control', []).append(float(power(sosfiltfilt(BRAIN, wv[cl], axis=-1)) / power(sosfiltfilt(BRAIN, x[cl], axis=-1))))
            d.setdefault('brain kept', []).append(float(np.mean([np.corrcoef(wv[i].ravel(), x[i].ravel())[0, 1] for i in cl])))
        for k in want:
            cnt[k] += 1
        cnt['clean'] += 1
    methods = list(per); keys = list(next(iter(per.values())))
    arrays = {}
    for k in keys:
        n = max(len(per[m].get(k, [])) for m in methods)
        arrays[k.replace(' ', '_')] = np.array([per[m].get(k, []) + [np.nan] * (n - len(per[m].get(k, []))) for m in methods])
    save_npz(ctx.numbers('filter', ctx.args.filter_source, f'atom_asr_TUAR{ctx.args.asr_tag}.npz'), methods=np.array(methods), keys=np.array(keys), cutoffs=np.array(ctx.args.asr_cutoffs),
             **arrays, **{k: np.array(v) for k, v in rem.items()})
    db = lambda a: float(np.maximum(10 * np.log10(np.maximum(np.nanmean(a), 1e-12)), DB_FLOOR))
    show = [('eyem__eye_left', 'eye'), ('musc__muscle_left', 'muscle'), ('elec__peak_left', 'peak'), ('line__line_left', 'line')]
    print('\n  TUAR patients per label: ' + ', '.join(f'{k} {cnt[k]}' for k in KINDS) + '; artifact change dB (control) | brain kept')
    for m in methods:
        print(f'    {m:18s} ' + ' '.join(f'{s} {db(arrays[k][methods.index(m)]):6.1f} ({db(arrays[k + "__control"][methods.index(m)]):5.1f})' for k, s in show if k in arrays) +
              f' | {np.nanmean(arrays["brain_kept"][methods.index(m)]):.3f}')
    print('\n  share of components removed (artifact windows / clean windows) per cutoff:')
    for c in ctx.args.asr_cutoffs:
        print(f'    {c:g}: ' + ', '.join(f'{k} {np.mean(rem[f"removed_{c:g}__{k}"]):.3f}' for k in (*KINDS, 'clean') if f'removed_{c:g}__{k}' in rem))


def draw(ctx):
    if not ctx.redraw:                                                      # numbers only, like the other noise panels
        return
    path = ctx.numbers('filter', ctx.args.filter_source, f'atom_asr_TUAR{ctx.args.asr_tag}.npz')
    if not os.path.exists(path):
        return
    z = load_npz(path); methods = [str(m) for m in z['methods']]
    cols = [(k, t) for k, t in (('eyem__eye_left', 'eye movement: frontal 1-10 Hz'), ('musc__muscle_left', 'muscle: 30-90 Hz'), ('elec__peak_left', 'electrode: peak amplitude'),
                                ('line__line_left', 'real line noise: 59-61 Hz')) if k in z]
    pal = {'backbone64': '#555555', 'multi-task 0.1': '#b3410f', 'atoms -eye': '#2a78d6', 'atoms -line': '#0b3d7a', 'ASR (time)': '#1baf7a'}
    colour = lambda m: pal.get(m, '#9a5fd0')
    db = lambda a: np.maximum(10 * np.log10(np.maximum(a, 1e-12)), DB_FLOOR)
    fig, ax = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 8.5), squeeze=False); x = np.arange(len(methods)); brain = np.nanmean(z['brain_kept'], axis=1)
    for j, (key, title) in enumerate(cols):
        art, ctl = np.nanmean(z[key], axis=1), np.nanmean(z[key + '__control'], axis=1); a = ax[0, j]
        for i, m in enumerate(methods):
            a.scatter(db(art[i]), brain[i], color=colour(m), s=70, marker='o' if m.startswith('ASR atoms') else 's', zorder=3, label=m if not m.startswith('ASR atoms') or m.endswith(' 5') else None)
        a.axvline(0, color='k', lw=.5, ls=':'); a.set_xlabel('artifact power change (dB)'); a.set_ylabel('brain kept on clean windows' if j == 0 else ''); a.set_title(title, fontsize=9, fontweight='bold')
        b = ax[1, j]; b.bar(x - .2, db(art), .38, color=[colour(m) for m in methods]); b.bar(x + .2, db(ctl), .38, color='#cccccc')
        b.axhline(0, color='k', lw=.6); b.set_xticks(x); b.set_xticklabels(methods, rotation=60, ha='right', fontsize=6); b.set_ylabel('power change (dB), grey: control' if j == 0 else '')
    ax[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle('ASR in the Q-atom space (purple, cutoff in the name) against atom removal and time-domain ASR, TUAR', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = ctx.figure_in('filter', ctx.args.filter_source, 'atom_asr_TUAR.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
