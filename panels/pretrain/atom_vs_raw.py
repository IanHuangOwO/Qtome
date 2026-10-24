"""atom_vs_raw: closed-form loso ridge on log Q-atom power vs on log raw band power, same trials and split -> analysis/atom_vs_raw.json.
Positive Q-atom - raw: the Q-atom code carries class information a raw spectral filterbank does not."""
import copy
import json

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

from cache_feature import CachedAtomDataset, get_atom_cache
from IO.splits import load_sessions
from panels.shared.ridge import ALPHAS, DATASETS
from train_finetune import apply_protocol


PRESETS = {'standard', 'quick'}


N_SEG = 4                                                   # time segments per trial (mean power in each)


BANDS_HZ = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def _loso_ridge(X, y, subj, keep, subs):
    """Closed-form ridge per held-out subject (alpha by leave-one-subject-out on the training
    subjects), features standardised on the training trials -> {subject: balanced accuracy}."""
    out = {}
    for s in subs:
        tr, te = keep & (subj != int(s)), keep & (subj == int(s))
        m, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = GridSearchCV(RidgeClassifier(class_weight='balanced'), {'alpha': ALPHAS}, cv=LeaveOneGroupOut(),
                           scoring='balanced_accuracy').fit((X[tr] - m) / sd, y[tr], groups=subj[tr])
        out[s] = float(balanced_accuracy_score(y[te], clf.predict((X[te] - m) / sd)))
    return out


def atom_vs_raw(config, checkpoint, out_path, datasets=DATASETS):
    """Per dataset, the same closed-form loso ridge on two fixed feature sets of the same trials:
    log Q-atom power (a^2 + b^2 per channel x Q-atom, mean over N_SEG time segments) and log raw band
    power (per channel x band, same segments). Q-atom - raw > 0: the Q-atom code carries class
    information a raw spectral filterbank does not."""
    from train_finetune import RawSource
    res = {}
    for ds, proto in datasets:
        ds_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds: ds_args}
        cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                              'split': {'type': 'loso'}}
        cfg['model_params'].setdefault('Qtome', {}).setdefault('finetune', {})
        cfg = apply_protocol(cfg)
        sessions = cfg['training_params']['finetune']['split'].get('sessions')
        subs = list(json.load(open(f"{ds_args['dataset_path']}/metadata.json"))['data_structure'])
        data = CachedAtomDataset(get_atom_cache(cfg, ds, subs), subs)
        raw = RawSource(cfg, ds, subs)
        assert torch.equal(data.labels, raw.labels), 'atom cache and raw trials are not in the same order'
        y, subj = data.labels.numpy(), data.subject_data.numpy()
        keep = np.isin(load_sessions(cfg, ds_args, subs, subj), sessions) if sessions is not None else np.ones(len(y), bool)
        amp = data.amp.float()                                                    # [T, N', Cv, S, 2]
        P = amp.pow(2).sum(-1)                                                    # [T, N', Cv, S]
        seg = torch.stack([c.mean(1) for c in P.tensor_split(min(N_SEG, P.shape[1]), dim=1)], 1)
        X_atom = (seg + 1e-6).log().reshape(len(y), -1).numpy()
        sf = float(cfg['preprocess_params']['sample_freq'])
        xr = raw.x.float()                                                        # [T, Cv, samples]
        spans = xr.tensor_split(N_SEG, dim=-1)
        nfft = max(spans[0].shape[-1], int(sf))                                   # >= 1 Hz bins: short spans (008, 1 s trials) leave no bin in 0.5-4 Hz
        f = torch.fft.rfftfreq(nfft, 1 / sf)
        bp = torch.stack([torch.stack([(torch.fft.rfft(sp, n=nfft, dim=-1).abs().pow(2)[..., (f >= lo) & (f < hi)]).mean(-1)
                                       for lo, hi in BANDS_HZ], -1) for sp in spans], 1)   # [T, seg, Cv, bands]
        X_raw = (bp + 1e-6).log().reshape(len(y), -1).numpy()
        st, rw = _loso_ridge(X_atom, y, subj, keep, subs), _loso_ridge(X_raw, y, subj, keep, subs)
        d = np.array([st[s] - rw[s] for s in subs])
        res[ds] = {'sessions': sessions, 'atom_power': float(np.mean(list(st.values()))),
                   'raw_band': float(np.mean(list(rw.values()))), 'diff': float(d.mean()),
                   'subjects_better': int((d > 0).sum()), 'n_subjects': len(subs),
                   'per_subject': {'atom_power': st, 'raw_band': rw}}
        print(f'  {ds:12s} loso ridge: atom power {res[ds]["atom_power"] * 100:5.1f} | raw band power '
              f'{res[ds]["raw_band"] * 100:5.1f} | atom - raw {d.mean() * 100:+.1f} ({(d > 0).sum()}/{len(subs)} subjects)')
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


def compute(ctx):
    atom_vs_raw(ctx.config, ctx.checkpoint, ctx.numbers('atom_vs_raw.json'))
