"""task_filter_zsubspace: noise removal in the encoder's output z (the [D] embedding of every channel-patch token that the qlet MLPs read) instead of on the
Qlet gains (a, b). For one artifact label, a low-rank direction set is fitted on TUAR patients that are NOT evaluated (even patient numbers fit, odd ones are
tested): the tokens of that label's windows against the tokens of the same patients' clean windows. Whiten by the clean covariance, take the top-k principal
directions of the artifact windows' (second moment about the clean mean) covariance there, and project them out of z of the test windows; the unchanged qlet
MLPs and templates then decode the projected z (the patch RMS is kept). Subspaces: eye (eyem windows), muscle (musc), electrode (elec) and artifacts (the five
labels pooled), at each rank in --zs-ranks. Scored like task_filter_artifacts on the held-out patients (artifact score and its clean-window control in dB,
brain kept = correlation with the input on clean windows), next to backbone64, the multi-task filter at 0.1 and the atom removals -eye and -line of the same run.
-> pretrain/analysis/filter/<source>/zsubspace_TUAR<tag>.npz (same keys as artifacts_TUAR.npz); figure only with --redraw."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import sosfiltfilt

from panels.core.numbers import load_npz, save_npz
from panels.pretrain.task_filter_artifacts import ATOM_GROUPS, BRAIN, COLUMNS, KINDS, LINE, SCORE_KIND, power, ratios, tuar_coverage
from panels.shared.task_filter.cells import FS
from panels.shared.task_filter.context import filter_context
from panels.shared.task_filter.tests import _setup, subjects

PRESETS = set()
SUBSPACES = {'eye': ('eyem',), 'muscle': ('musc',), 'electrode': ('elec',), 'artifacts': KINDS}
BASELINES = ('backbone64', 'multi-task 0.1', 'atoms -eye', 'atoms -line')
DB_FLOOR = -40.0


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--art-patients', type=int, default=8, help='task_filter_artifacts: TUAR patients wanted per artifact type (patients tried in a spread-out order)')
    flags.add('--art-windows', type=int, default=20, help='task_filter_artifacts: windows per artifact type and patient (clean: 1.5 x this)')
    flags.add('--zs-ranks', type=int, nargs='+', default=[2, 8, 24], help='task_filter_zsubspace: number of z directions removed')
    flags.add('--zs-ridge', type=float, default=1e-3, help='task_filter_zsubspace: ridge on the clean covariance, as a share of its mean eigenvalue (large = directions ranked by artifact energy, small = by artifact / clean ratio)')
    flags.add('--zs-tag', default='', help='task_filter_zsubspace: suffix of the output file, to keep variants apart')
    flags.add('--zs-fit-patients', type=int, default=8, help='task_filter_zsubspace: TUAR patients (even numbers) the subspaces are fitted on')


class Moments:
    """Running count, sum and second moment about a fixed centre of z tokens."""
    def __init__(self, d):
        self.n, self.s, self.m2 = 0, np.zeros(d), np.zeros((d, d))

    def add(self, tok, centre=None):
        tok = tok.astype(np.float64); self.n += len(tok); self.s += tok.sum(0)
        c = tok if centre is None else tok - centre
        self.m2 += c.T @ c


def dst_windows(f, nw, dst_r, dst_c):
    """One TUAR patient -> (patient, raw [N, C, T] of the montage's channels, {label: window indices}) or None when the annotation layout does not match."""
    z = np.load(f); T = int(z['valid_end'][0]); sub = int(os.path.basename(f).split('_')[0])
    raw = z['data'].astype(np.float64)[:, [dst_r.index(n) for n in dst_c], :T]
    cov, anyc = tuar_coverage(sub)
    if len(anyc) != len(raw) or not np.array_equal((anyc >= 0.5).astype(int), z['labels']):
        return None
    rng = np.random.default_rng(sub)
    idx = {k: rng.permutation(np.flatnonzero((cov[k] >= 0.5) & (anyc - cov[k] < 0.1)))[:nw] for k in KINDS}
    idx['clean'] = rng.permutation(np.flatnonzero(anyc == 0))[:int(1.5 * nw)]
    return sub, raw, idx


def stack(raw, idx, sets):
    pos, allidx = {}, []
    for k in sets:
        pos[k] = np.arange(len(allidx), len(allidx) + len(idx[k])); allidx += list(idx[k])
    allidx = np.array(allidx); x = raw[allidx]
    return pos, x / x.std((1, 2), keepdims=True)


def fit_directions(art, clean, ridge):
    """-> (mean, Wh, Whinv, Q [D, D] eigenvectors of the whitened artifact covariance, strongest first, their eigenvalues)."""
    mu = clean.s / clean.n; cov_c = clean.m2 / clean.n; cov_a = art.m2 / art.n
    w, v = np.linalg.eigh(cov_c + np.eye(len(mu)) * ridge * np.trace(cov_c) / len(mu))
    wh, whinv = (v * w ** -0.5) @ v.T, (v * w ** 0.5) @ v.T
    lam, q = np.linalg.eigh(wh @ cov_a @ wh)
    return mu, wh, whinv, q[:, ::-1], lam[::-1]


def projector(fit, k):
    mu, wh, whinv, q, _ = fit
    return mu, wh @ (np.eye(len(mu)) - q[:, :k] @ q[:, :k].T) @ whinv


def compute(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    columns = [(k, sc) for k, sc in COLUMNS if (k is None or k in KINDS) and sc != 'pair left']
    dst_r, dst_c, sel, encode, ops = _setup(tf, 'TUAR'); dec = tf.backbone.decode; model = tf.backbone.model.eval()
    up = [n.upper() for n in dst_c]; fp = [up.index(n) for n in ('FP1', 'FP2')]
    zl = load_npz(ctx.numbers('atom_labels.npz')); top = [str(zl['classes'][i]) for i in zl['probs'].argmax(1)]
    atom_keep = {n: torch.tensor([t not in ATOM_GROUPS[n] for t in top], dtype=torch.float32) for n in ('atoms -eye', 'atoms -line')}
    files = subjects('TUAR'); order = [files[i] for i in np.argsort((np.arange(len(files)) * 0.6180339887) % 1)[:150]]
    nw = ctx.args.art_windows; D = model.atoms.input_norm.normalized_shape[0]

    mom = {k: Moments(D) for k in (*KINDS, 'clean')}; fitted = 0                          # pass 1: tokens of the fit patients (even numbers)
    for f in order:
        if fitted >= ctx.args.zs_fit_patients:
            break
        if int(os.path.basename(f).split('_')[0]) % 2:
            continue
        w = dst_windows(f, nw, dst_r, dst_c)
        if w is None or len(w[2]['clean']) < 8:
            continue
        sub, raw, idx = w; pos, x = stack(raw, idx, (*KINDS, 'clean'))
        _, z = encode(x, True, want_z=True); z = z[:, :, sel].numpy()                    # [N, P, C, D]
        print(f'  fit patient {sub}: ' + ', '.join(f'{k} {len(pos[k])}' for k in pos))
        for k in pos:
            if len(pos[k]):
                if k == 'clean':
                    mom[k].add(z[pos[k]].reshape(-1, D))
        mu_c = mom['clean'].s / mom['clean'].n
        for k in KINDS:
            if len(pos[k]) >= 5:
                mom[k].add(z[pos[k]].reshape(-1, D), centre=mu_c)
        fitted += 1
    assert fitted and mom['clean'].n, 'no fit patient (even number) with clean windows'
    clean = mom['clean']; clean.m2 -= np.outer(clean.s, clean.s) / clean.n                 # centred second moment of the clean tokens
    fits = {}
    for name, labels in SUBSPACES.items():
        art = Moments(D); art.n = sum(mom[k].n for k in labels); art.m2 = sum(mom[k].m2 for k in labels)
        if art.n:
            fits[name] = fit_directions(art, clean, ctx.args.zs_ridge)
            print(f'  subspace {name}: {art.n} artifact tokens, top whitened eigenvalues ' + ', '.join(f'{v:.1f}' for v in fits[name][4][:5]))
    mu_all = clean.s / clean.n

    per, cnt = {}, {k: 0 for k in (*KINDS, 'clean')}                                     # pass 2: the held-out patients (odd numbers)
    for f in order:
        if all(cnt[k] >= ctx.args.art_patients for k in KINDS):
            break
        if not int(os.path.basename(f).split('_')[0]) % 2:
            continue
        w = dst_windows(f, nw, dst_r, dst_c)
        if w is None:
            continue
        sub, raw, idx = w
        want = [k for k in KINDS if len(idx[k]) >= 5 and cnt[k] < ctx.args.art_patients]
        if len(idx['clean']) < 8 or not want:
            continue
        pos, x = stack(raw, idx, (*want, 'clean')); T = x.shape[-1]; cl = pos['clean']
        g, z = encode(x, True, want_z=True); B, N = z.shape[:2]
        amp0 = model.atoms.dense_amp(z.reshape(B * N, 64, D)).reshape(B, N, 64, -1, 2)
        scale = g.flatten(3).norm(dim=-1) / amp0.flatten(3).norm(dim=-1).clamp_min(1e-12)    # the patch RMS (and the trial scale) of every token

        def through(zz):
            a = model.atoms.dense_amp(zz.reshape(B * N, 64, D)).reshape(B, N, 64, -1, 2) * scale[..., None, None]
            return dec(a[:, :, sel])[..., :T]
        base = dec(g[:, :, sel])[..., :T]
        assert np.corrcoef(through(z).ravel(), base.ravel())[0, 1] > 0.999, 'decoding the unchanged z does not give backbone64'
        sig = {'backbone64': base}
        if 0.1 in ops:
            sig['multi-task 0.1'] = dec(torch.einsum('scd,bndsx->bncsx', ops[0.1], g))[..., :T]
        for n, m in atom_keep.items():
            sig[n] = dec(g[:, :, sel] * m.view(1, 1, 1, -1, 1))[..., :T]
        for name, fit in fits.items():
            for k in ctx.args.zs_ranks:
                mu, M = projector(fit, k)
                sig[f'z -{name} {k}'] = through((z - torch.tensor(mu).float()) @ torch.tensor(M).float() + torch.tensor(mu).float())
        print(f'  test patient {sub}: ' + ', '.join(f'{k} {len(pos[k])}' for k in pos))
        for m, wv in sig.items():
            d = per.setdefault(m, {})
            for k, score in columns:
                if k in want:
                    d.setdefault(f'{k}__{score}', []).append(float(ratios(wv, x, pos[k], SCORE_KIND[score], fp)))
                    d.setdefault(f'{k}__{score}__control', []).append(float(ratios(wv, x, cl, SCORE_KIND[score], fp)))
            # real line noise: 59-61 Hz power of the clean windows, control = 8-30 Hz
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
    save_npz(ctx.numbers('filter', ctx.args.filter_source, f'zsubspace_TUAR{ctx.args.zs_tag}.npz'), methods=np.array(methods), keys=np.array(keys), ranks=np.array(ctx.args.zs_ranks), **arrays)
    db = lambda a: float(np.maximum(10 * np.log10(np.maximum(np.nanmean(a), 1e-12)), DB_FLOOR))
    show = [('eyem__eye_left', 'eye'), ('musc__muscle_left', 'muscle'), ('elec__peak_left', 'peak'), ('line__line_left', 'line')]
    print(f'\n  held-out TUAR patients per label: ' + ', '.join(f'{k} {cnt[k]}' for k in KINDS) + '; artifact change dB (control) | brain kept')
    for m in methods:
        print(f'    {m:18s} ' + ' '.join(f'{s} {db(arrays[k][methods.index(m)]):6.1f} ({db(arrays[k + "__control"][methods.index(m)]):5.1f})' for k, s in show if k in arrays) +
              f' | {np.nanmean(arrays["brain_kept"][methods.index(m)]):.3f}')


def draw(ctx):
    if not ctx.redraw:                                                      # numbers only, like the other noise panels
        return
    path = ctx.numbers('filter', ctx.args.filter_source, f'zsubspace_TUAR{ctx.args.zs_tag}.npz')
    if not os.path.exists(path):
        return
    z = load_npz(path); methods = [str(m) for m in z['methods']]
    cols = [(k, t) for k, t in (('eyem__eye_left', 'eye movement: frontal 1-10 Hz'), ('musc__muscle_left', 'muscle: 30-90 Hz'), ('elec__peak_left', 'electrode: peak amplitude'),
                                ('line__line_left', 'real line noise: 59-61 Hz')) if k in z]
    pal = {'backbone64': '#555555', 'multi-task 0.1': '#b3410f', 'atoms -eye': '#2a78d6', 'atoms -line': '#0b3d7a', 'z -eye': '#d62728', 'z -muscle': '#17becf',
           'z -electrode': '#8c564b', 'z -artifacts': '#9a5fd0'}
    colour = lambda m: pal.get(m.rsplit(' ', 1)[0] if m.startswith('z ') else m, '#888888')
    db = lambda a: np.maximum(10 * np.log10(np.maximum(a, 1e-12)), DB_FLOOR)
    fig, ax = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 8.5), squeeze=False); x = np.arange(len(methods))
    brain = np.nanmean(z['brain_kept'], axis=1)
    for j, (key, title) in enumerate(cols):
        art, ctl = np.nanmean(z[key], axis=1), np.nanmean(z[key + '__control'], axis=1)
        a = ax[0, j]
        for i, m in enumerate(methods):
            a.scatter(db(art[i]), brain[i], color=colour(m), s=25 + 6 * int(m.rsplit(' ', 1)[1]) if m.startswith('z ') else 70, marker='o' if m.startswith('z ') else 's', zorder=3, label=m.rsplit(' ', 1)[0] if m.startswith('z ') and m.endswith(' 2') else (m if not m.startswith('z ') else None))
        a.axvline(0, color='k', lw=.5, ls=':'); a.set_xlabel('artifact power change (dB)'); a.set_ylabel('brain kept on clean windows' if j == 0 else ''); a.set_title(title, fontsize=9, fontweight='bold')
        b = ax[1, j]; b.bar(x - .2, db(art), .38, color=[colour(m) for m in methods]); b.bar(x + .2, db(ctl), .38, color='#cccccc')
        b.axhline(0, color='k', lw=.6); b.set_xticks(x); b.set_xticklabels(methods, rotation=60, ha='right', fontsize=6); b.set_ylabel('power change (dB), grey: control' if j == 0 else '')
    ax[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Removal in the z subspace (dot size = number of directions removed) against atom removal, held-out TUAR patients', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = ctx.figure_in('filter', ctx.args.filter_source, 'zsubspace_TUAR.png'); fig.savefig(out, dpi=110); plt.close(fig); print(f'  -> {out}')
