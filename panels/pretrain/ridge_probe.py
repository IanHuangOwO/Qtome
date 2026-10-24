"""ridge_probe: deterministic closed-form linear probe on the frozen pre-atom z, Compass LOSO on BNCI2014004 / 001 / 008 ->
analysis/ridge_probe.json (the latent feature cache is built on first use). A deterministic readout of how linearly usable the
representation is, without SGD head training (the finetune heads' run-to-run noise).

Per dataset, Compass LOSO split (the protocol's session selection, configs/finetune_protocols.json incl. _dataset_split, Qtome heads
in configs/Qtome/protocol_heads.json). Per held-out subject: PCA of the training trials' z tokens (D -> n_pca, like latent_proj
'pca'), flatten [patches, channels, n_pca], standardise on the training trials, ridge classifier (balanced class weights; alpha by
leave-one-subject-out over the training subjects). Score: balanced accuracy on the held-out subject, mean over subjects. Same inputs
-> same number: differences between backbones are not head-training noise. A relative ruler between backbones, not a Compass
number: on BNCI2014001 it reads ~10 points below the factorised SGD probe (latent_signed), whose low-rank form regularises
22-channel data better than plain L2.

fewshot_ridge (a few-shot variant on the calibration trials, cited in docs/finetune-caveats.md) is kept here; no panel runs it."""
import copy
import json

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

from model.Qtome.finetune.feature_cache import CachedAtomDataset, get_atom_cache
from IO.splits import load_sessions, make_runs
from panels.shared.ridge import ALPHAS, DATASETS, _features, _pca_axes
from model.Qtome.finetune.protocol import apply_protocol


PRESETS = {'standard', 'quick'}


def ridge_probe(config, checkpoint, out_path, n_pca=8, datasets=DATASETS, pool=1, feature='z'):
    """pool > 1: average `pool` adjacent z tokens first (train_finetune.pool_tokens). feature: 'z' or
    'atom_hidden' (the probe reads every Q-atom's MLP hidden instead of z; same PCA pipeline)."""
    from model.Qtome.finetune.sources import pool_tokens
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
        data = CachedAtomDataset(get_atom_cache(cfg, ds, subs, latent='output'), subs)
        z, y = _features(feature, data, checkpoint), data.labels.numpy()   # [T, N', Cv, D]
        if pool > 1:
            z = pool_tokens(z, pool)
        subj = data.subject_data.numpy()
        keep = np.ones(len(y), bool)
        if sessions is not None:
            keep = np.isin(load_sessions(cfg, ds_args, subs, subj), sessions)
        T, Np, Cv, D = z.shape
        scores = {}
        for s in subs:
            tr, te = keep & (subj != int(s)), keep & (subj == int(s))
            mu, W = _pca_axes(z[torch.from_numpy(tr)].reshape(-1, D), n_pca)
            f = ((z - mu) @ W).reshape(T, -1).numpy()                                # [T, N' * Cv * n_pca]
            m, sd = f[tr].mean(0), f[tr].std(0) + 1e-6
            # alpha by leave-one-subject-out on the training subjects (the test is cross-subject; a
            # leave-one-trial-out choice rewards subject-specific fit)
            clf = GridSearchCV(RidgeClassifier(class_weight='balanced'), {'alpha': ALPHAS}, cv=LeaveOneGroupOut(),
                               scoring='balanced_accuracy').fit((f[tr] - m) / sd, y[tr], groups=subj[tr])
            scores[s] = float(balanced_accuracy_score(y[te], clf.predict((f[te] - m) / sd)))
        res[ds] = {'split': 'loso', 'sessions': sessions, 'per_subject': scores,
                   'mean': float(np.mean(list(scores.values())))}
        print(f'  {ds:12s} loso ridge probe: {res[ds]["mean"] * 100:5.1f}  (sessions {sessions}; '
              + ' '.join(f'{k}:{v * 100:.0f}' for k, v in scores.items()) + ')')
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


FEWSHOT = (('BNCI2014004', 'mi_fewshot'), ('BNCI2014001', 'mi_fewshot'), ('BNCI2014008', 'p300_fewshot'))


def _segment_means(t, n_seg):
    """t [T, N', ...] -> [T, n_seg * ...]: mean over n_seg consecutive stretches of the token axis, flattened."""
    return torch.stack([c.mean(1) for c in t.tensor_split(min(n_seg, t.shape[1]), dim=1)], 1).reshape(len(t), -1)


def fewshot_ridge(config, checkpoint, out_path, n_pca=8, n_seg=4, datasets=FEWSHOT, feature='z'):
    """Within-subject few-shot (the protocol's chronological calibration split, per subject) with a closed-form
    probe instead of a trained head: RidgeClassifierCV, its shrinkage picked by efficient leave-one-out on the
    subject's calibration trials only. Two compact fixed feature sets per trial: 'z' = z tokens projected on
    n_pca principal axes (fit on the calibration tokens), averaged over n_seg time segments; 'z_power' = log mean
    square of the same projections per segment (power, what MI carries); 'Q-atom' = log Q-atom
    power (a^2 + b^2) averaged over the same segments. -> {dataset: {feature: {mean, per_subject}}}."""
    from sklearn.linear_model import RidgeClassifierCV
    res = {}
    for ds, proto in datasets:
        ds_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds: ds_args}
        cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                              'split': {'type': 'fewshot', 'train_fraction': 0.3}}
        cfg['model_params'].setdefault('Qtome', {}).setdefault('finetune', {})
        cfg = apply_protocol(cfg)
        split = cfg['training_params']['finetune']['split']
        subs = list(json.load(open(f"{ds_args['dataset_path']}/metadata.json"))['data_structure'])
        data = CachedAtomDataset(get_atom_cache(cfg, ds, subs, latent='output'), subs)
        z, y, subj = _features(feature, data, checkpoint), data.labels.numpy(), data.subject_data.numpy()
        atom = _segment_means((data.amp.float().pow(2).sum(-1) + 1e-6).log(), n_seg).numpy()
        runs = make_runs(split, subs, subj, y, load_sessions(cfg, ds_args, subs, subj))
        scores = {'z': {}, 'z_power': {}, 'atom': {}}
        for run in runs:
            tr = run['train']
            (s, te), = run['eval']['heldout'].items()
            mu, W = _pca_axes(z[torch.from_numpy(tr)].reshape(-1, z.shape[-1]), n_pca)
            proj = (z[np.concatenate([tr, te])] - mu) @ W                              # [T, N', Cv, n_pca]
            feats = {'z': _segment_means(proj, n_seg).numpy(),
                     'z_power': _segment_means(proj.pow(2), n_seg).add(1e-6).log().numpy(),
                     'atom': atom[np.concatenate([tr, te])]}
            for k, f in feats.items():
                ftr, fte = f[:len(tr)], f[len(tr):]
                m, sd = ftr.mean(0), ftr.std(0) + 1e-6
                clf = RidgeClassifierCV(alphas=ALPHAS, class_weight='balanced').fit((ftr - m) / sd, y[tr])
                scores[k][s] = float(balanced_accuracy_score(y[te], clf.predict((fte - m) / sd)))
        res[ds] = {k: {'mean': float(np.mean(list(v.values()))), 'per_subject': v} for k, v in scores.items()}
        res[ds]['split'] = split
        print(f"  {ds:12s} few-shot closed-form ridge: z {res[ds]['z']['mean'] * 100:5.1f} | z log-power "
              f"{res[ds]['z_power']['mean'] * 100:5.1f} | atom power {res[ds]['atom']['mean'] * 100:5.1f}"
              f"  (train_fraction {split.get('train_fraction')})")
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


def compute(ctx):
    ridge_probe(ctx.config, ctx.checkpoint, ctx.numbers('ridge_probe.json'))
