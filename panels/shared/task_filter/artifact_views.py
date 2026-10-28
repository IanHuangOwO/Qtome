"""The extra views of task_filter_artifacts (all from the numbers it saves, nothing recomputed): frequency profile of the removal, dose-response, per-atom
picture, artifact decodability, scalp maps, example traces and per-patient scores. Every function draws one figure from the loaded numbers `z`
and returns its path."""
import numpy as np
import matplotlib.pyplot as plt

FS = 200
LABELS = ('eyem', 'musc', 'elec', 'eyem_musc', 'musc_elec')
NAME = {'eyem': 'eye movement (eyem)', 'musc': 'muscle (musc)', 'elec': 'electrode (elec)', 'eyem_musc': 'mixed eyem_musc', 'musc_elec': 'mixed musc_elec'}
SETS = (*LABELS, 'clean')
QUANT = {'eyem': 0, 'musc': 1, 'elec': 2}                                     # which stored per-window quantity a score reads: eye, muscle, peak
SCORE_SET = (('eyem', 'eye'), ('musc', 'muscle'), ('elec', 'peak'), ('eyem_musc', 'eye'), ('eyem_musc', 'muscle'), ('musc_elec', 'muscle'), ('musc_elec', 'peak'))
SCORE_Q = {'eye': 0, 'muscle': 1, 'peak': 2}
FLOOR = -40.0
san = lambda m: m.replace(' ', '_')
db = lambda r: np.maximum(10 * np.log10(np.maximum(r, 1e-12)), FLOOR)


def _sel(z, k):
    return np.flatnonzero(z['win_set'] == SETS.index(k))


def _patients_with(z, k):
    return sorted(set(z['win_patient'][_sel(z, k)].tolist()))


def spectrum(z, shown, colors, path):
    """Output / input power against frequency, per label: on the label's windows and on the clean windows of the same patients (the control)."""
    fig, ax = plt.subplots(2, len(LABELS), figsize=(4.2 * len(LABELS), 8), squeeze=False)
    fr = np.arange(101)
    for j, k in enumerate(LABELS):
        pats = _patients_with(z, k)
        if not pats:
            continue
        for r, which in enumerate((k, 'clean')):
            for m in shown:
                curves = []
                for p in pats:
                    w = np.flatnonzero((z['win_set'] == SETS.index(which)) & (z['win_patient'] == p))
                    if len(w):
                        curves.append(db(z[f'psd__{san(m)}'][w].mean(0) / z['psd__input'][w].mean(0)))
                c = np.array(curves)
                if not len(c):
                    continue
                mu, se = c.mean(0), (c.std(0, ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0 * c[0])
                ax[r, j].plot(fr[1:], mu[1:], color=colors[m], lw=1.6, label=m); ax[r, j].fill_between(fr[1:], (mu - se)[1:], (mu + se)[1:], color=colors[m], alpha=.15, lw=0)
            ax[r, j].axhline(0, color='k', lw=.6); ax[r, j].axvline(60, color='#999999', lw=.6, ls=':'); ax[r, j].set_ylim(FLOOR - 1, 6); ax[r, j].set_xlim(1, 100)
            ax[r, j].set_xlabel('Hz'); ax[r, j].set_ylabel('output power / input power (dB)')
            ax[r, j].set_title((f'{NAME[k]}: {len(pats)} patients\nartifact windows' if r == 0 else 'control: clean windows of the same patients'), fontsize=9, fontweight='bold' if r == 0 else 'normal')
    ax[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Where each method removes power: output / input power against frequency (dotted: 60 Hz; floor -40 dB)', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def dose(z, shown, colors, path):
    """Output / input power of one window against how strong the artifact is in it (input power over the patient's clean-window median, dB)."""
    fig, ax = plt.subplots(1, len(SCORE_SET), figsize=(3.6 * len(SCORE_SET), 4.4), squeeze=False)
    for j, (k, sc) in enumerate(SCORE_SET):
        qi = SCORE_Q[sc]; w = _sel(z, k)
        if not len(w):
            continue
        base = {p: np.median(z['q__input'][(z['win_set'] == SETS.index('clean')) & (z['win_patient'] == p), qi]) for p in set(z['win_patient'][w].tolist())}
        strength = 10 * np.log10(z['q__input'][w, qi] / np.array([base[p] for p in z['win_patient'][w]]))
        edges = np.quantile(strength, np.linspace(0, 1, 5)); bins = np.clip(np.searchsorted(edges, strength, side='right') - 1, 0, 3)
        for m in shown:
            y = db(z[f'q__{san(m)}'][w, qi] / z['q__input'][w, qi])
            mu = [y[bins == b].mean() for b in range(4)]; se = [y[bins == b].std(ddof=1) / np.sqrt((bins == b).sum()) if (bins == b).sum() > 1 else 0 for b in range(4)]
            ax[0, j].errorbar([strength[bins == b].mean() for b in range(4)], mu, yerr=se, color=colors[m], marker='o', ms=4, lw=1.4, capsize=2, label=m)
        ax[0, j].axhline(0, color='k', lw=.6); ax[0, j].set_xlabel('artifact strength (dB over clean median)', fontsize=8); ax[0, j].set_ylabel('output / input (dB)')
        ax[0, j].set_title(f'{NAME[k]}\n{sc} score, {len(w)} windows', fontsize=9, fontweight='bold')
    ax[0, 0].legend(fontsize=7)
    fig.suptitle('Does a method remove strong artifacts more than weak ones? x: the window\'s score quantity over the same patient\'s clean median (4 equal-count bins)', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def atoms(z, shares, path):
    """Per atom (ordered by peak frequency): how much its power rises in the label's windows (Cohen d against the same patients' clean windows, log power,
    centred per patient), and how much of it each multi-task share keeps (output / input power, dB)."""
    S = z['atom_in'].shape[1]; order = np.argsort(z['atom_peak'], kind='stable'); lg = np.log(z['atom_in'] + 1e-20)
    cen = lg.copy()
    for p in set(z['win_patient'].tolist()):
        c = (z['win_patient'] == p) & (z['win_set'] == SETS.index('clean'))
        if c.any():
            cen[z['win_patient'] == p] -= lg[c].mean(0)
    fig, ax = plt.subplots(2, len(LABELS), figsize=(4.6 * len(LABELS), 8), squeeze=False)
    cols = {sh: c for sh, c in zip(shares, ('#eb6834', '#6d1d05'))}
    for j, k in enumerate(LABELS):
        a, c = cen[_sel(z, k)], cen[_sel(z, 'clean')]
        pats = set(z['win_patient'][_sel(z, k)].tolist()); c = cen[(z['win_set'] == SETS.index('clean')) & np.isin(z['win_patient'], list(pats))]
        if len(a) < 3:
            continue
        sd = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(c) - 1) * c.var(0, ddof=1)) / (len(a) + len(c) - 2)); d = (a.mean(0) - c.mean(0)) / np.maximum(sd, 1e-9)
        ax[0, j].bar(np.arange(S), d[order], color=np.where(d[order] > 0, '#2a78d6', '#888888'))
        ax[0, j].axhline(0, color='k', lw=.6); ax[0, j].set_ylabel('Cohen d, atom power: artifact vs clean')
        ax[0, j].set_title(f'{NAME[k]}\n{len(a)} windows, {len(pats)} patients', fontsize=9, fontweight='bold')
        w = _sel(z, k)
        for sh in shares:
            r = 10 * np.log10(z[f'atom_out__{san(sh)}'][w].sum(0) / z['atom_in'][w].sum(0))
            ax[1, j].plot(np.arange(S), np.maximum(r[order], FLOOR), 'o-', ms=3, lw=1.2, color=cols[sh], label=sh)
        ax[1, j].axhline(0, color='k', lw=.6); ax[1, j].set_ylabel('power kept by the filter (dB)'); ax[1, j].set_ylim(FLOOR - 1, 5)
        for a_ in (ax[0, j], ax[1, j]):
            a_.set_xticks(np.arange(S)[::2]); a_.set_xticklabels([f'{z["atom_peak"][order][i]:.0f}' for i in range(0, S, 2)], fontsize=6, rotation=90); a_.set_xlabel('atom, by peak frequency (Hz)')
    ax[1, 0].legend(fontsize=7)
    fig.suptitle('Which atoms carry each artifact (top) and which the multi-task filter keeps (bottom): an atom high in the top row and near 0 dB below is an artifact atom the filter keeps', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def decode(z, shown, colors, path):
    """Cross-patient AUC of a logistic classifier (standardised log band power per channel, 8 bands) separating each label's windows from the clean windows of the
    same patients, trained and tested on a method's own output: how much artifact information the output still carries (0.5 = none)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    methods = ['input', *shown]
    fig, ax = plt.subplots(figsize=(11, 4.6)); res = {}
    for k in LABELS:
        pats = _patients_with(z, k); idx = np.flatnonzero(np.isin(z['win_patient'], pats) & np.isin(z['win_set'], [SETS.index(k), SETS.index('clean')]))
        if len(pats) < 3:
            continue
        y = (z['win_set'][idx] == SETS.index(k)).astype(int); g = z['win_patient'][idx]
        for m in methods:
            X = z[f'feat__{san(m)}'][idx]; score = np.zeros(len(idx))
            for tr, te in GroupKFold(n_splits=min(5, len(pats))).split(X, y, g):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000)).fit(X[tr], y[tr])
                score[te] = clf.decision_function(X[te])
            res[(k, m)] = roc_auc_score(y, score)
    ks = [k for k in LABELS if (k, 'input') in res]; w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(ks)) + (i - (len(methods) - 1) / 2) * w, [res[(k, m)] for k in ks], w, color=colors.get(m, '#000000'), label=m)
    ax.axhline(0.5, color='k', lw=.8, ls=':'); ax.set_ylim(0.4, 1.02); ax.set_xticks(np.arange(len(ks))); ax.set_xticklabels([NAME[k] for k in ks]); ax.set_ylabel('cross-patient AUC, artifact vs clean')
    ax.legend(fontsize=7, ncol=len(methods)); ax.set_title('How much artifact information is left in each method\'s output (lower = more removed; 0.5 = none; classifier retrained on each output)', fontsize=10, fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path, res


def maps(z, shares_m, path):
    """Scalp maps per label: where the artifact adds power (input, artifact vs the same patients' clean windows, dB) and where methods remove it (output / input, dB)."""
    from panels.kit.topomap import build_triangulation, draw_topomap, project_coords_2d
    pos = project_coords_2d(z['coords']); tri = build_triangulation(pos)
    rows = (('input: artifact power over clean (dB)', None), *[(f'{m}: output / input power (dB)', m) for m in shares_m])
    fig, ax = plt.subplots(len(rows), len(LABELS), figsize=(3.3 * len(LABELS), 3.0 * len(rows)), squeeze=False)
    for j, k in enumerate(LABELS):
        w = _sel(z, k); pats = list(set(z['win_patient'][w].tolist()))
        c = np.flatnonzero((z['win_set'] == SETS.index('clean')) & np.isin(z['win_patient'], pats))
        if not len(w):
            continue
        for r, (title, m) in enumerate(rows):
            if m is None:
                v = 10 * np.log10(z['chan__input'][w].mean(0) / z['chan__input'][c].mean(0))
            else:
                v = 10 * np.log10(z[f'chan__{san(m)}'][w].mean(0) / z['chan__input'][w].mean(0))
            lim = max(np.abs(v).max(), 1e-3); draw_topomap(ax[r, j], pos, v, cmap='RdBu_r', vmin=-lim, vmax=lim, triang=tri)
            ax[r, j].set_title(f'{NAME[k]}\n{title}' if r == 0 else title, fontsize=8, fontweight='bold' if r == 0 else 'normal')
            ax[r, j].text(0.5, -0.05, f'{v.min():+.1f} to {v.max():+.1f} dB', ha='center', fontsize=7, transform=ax[r, j].transAxes)
    fig.suptitle('Scalp maps (each map has its own symmetric colour scale; the range is under it)', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def examples(z, show, colors, path):
    """One window per label (the median one of the first patient that has it), three channels, in microvolts. Per channel two rows on the same vertical scale:
    the input, and what each method removed from it (input minus output; a flat line = nothing removed)."""
    names = [str(n) for n in z['ex_channels']]; ks = [k for k in LABELS if f'example__{k}__input' in z]
    fig, ax = plt.subplots(2 * len(names), len(ks), figsize=(4.2 * len(ks), 1.7 * 2 * len(names)), squeeze=False, sharex=True); t = np.arange(1000) / FS
    for j, k in enumerate(ks):
        for r, n in enumerate(names):
            x = z[f'example__{k}__input'][r]; lim = 1.1 * np.abs(x).max()
            top, bot = ax[2 * r, j], ax[2 * r + 1, j]
            top.plot(t, x, color='k', lw=0.8); bot.axhline(0, color='k', lw=.4)
            for m in show:
                bot.plot(t, x - z[f'example__{k}__{san(m)}'][r], color=colors.get(m, '#888888'), lw=0.8, label=m, alpha=.9)
            for a in (top, bot):
                a.set_ylim(-lim, lim)
            top.set_ylabel(f'{n}\ninput (uV)', fontsize=7); bot.set_ylabel('removed (uV)', fontsize=7)
            if r == 0:
                top.set_title(NAME[k], fontsize=9, fontweight='bold')
        ax[-1, j].set_xlabel('s')
    ax[1, 0].legend(fontsize=6, ncol=2, loc='upper right')
    fig.suptitle('One window per label: input (black) and what each method removed from it (input minus output, same scale as the input)', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def patients(z, shown, colors, keys, path):
    """Per-patient scores behind the error bars: one dot per patient (dB), the mean as a bar; one panel per score column."""
    fig, ax = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.2), squeeze=False); rng = np.random.default_rng(0); methods = [str(m) for m in z['methods']]
    for j, (key, title) in enumerate(keys):
        a = ax[0, j]
        for i, m in enumerate(shown):
            v = db(z[key.replace(' ', '_')][methods.index(m)]); v = v[~np.isnan(v)]
            a.bar(i, v.mean(), .6, color=colors.get(m, '#888888'), alpha=.45); a.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=14, color='k', zorder=3)
        a.axhline(0, color='k', lw=.6); a.set_xticks(range(len(shown))); a.set_xticklabels(shown, rotation=40, ha='right', fontsize=7); a.set_ylabel('output / input power (dB)'); a.set_title(title, fontsize=8, fontweight='bold')
    fig.suptitle('Per-patient scores of the main figure (dots = patients, bars = mean)', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=110); plt.close(fig)
    return path
