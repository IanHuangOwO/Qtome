"""coord_robustness: loso ridge probe trained on normal coordinates, tested with jittered / mirrored coordinates -> analysis/coord_robustness.json."""
import copy
import json

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

from cache_feature import CachedAtomDataset, get_atom_cache
from IO.splits import load_sessions
from panels.shared.ridge import ALPHAS, _pca_axes
from train_finetune import apply_protocol



COORD_TRANSFORMS = ('jitter_2mm', 'jitter_5mm', 'jitter_10mm', 'mirror')


def coord_robustness(config, checkpoint, out_path, n_pca=8, transforms=COORD_TRANSFORMS,
                     datasets=(('BNCI2014004', 'mi_loso'), ('BNCI2014001', 'mi_loso'))):
    """Coordinate robustness: the loso ridge probe (as ridge_probe) trained on the training
    subjects' normal-coordinate z, tested on the held-out subject's z extracted with transformed coordinates
    (cache_feature.transform_coords, data untouched). Per transform: balanced accuracy, the fraction of held-out
    predictions that change vs normal coordinates, and the fraction of hand trials predicted as the other hand."""
    res = {}
    for ds, proto in datasets:
        base_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        meta = json.load(open(f"{base_args['dataset_path']}/metadata.json"))
        subs = list(meta['data_structure'])
        tg = meta['data_metadata']['targets']
        hand = {side: next(int(k) for k, v in tg.items() if k.isdigit() and side in json.dumps(v).lower())
                for side in ('left', 'right')}

        def load(tf):
            args = dict(base_args, **({'coords_transform': tf} if tf else {}))
            cfg = copy.deepcopy(config)
            cfg['dataset_params']['finetune'] = {ds: args}
            cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                                  'split': {'type': 'loso'}}
            cfg['model_params'].setdefault('Qtome', {}).setdefault('finetune', {})
            cfg = apply_protocol(cfg)
            data = CachedAtomDataset(get_atom_cache(cfg, ds, subs, latent='output'), subs)
            return cfg, args, data

        cfg, args, data = load(None)
        z, y, subj = data.z.float(), data.labels.numpy(), data.subject_data.numpy()
        zt = {tf: load(tf)[2].z.float() for tf in transforms}
        sessions = cfg['training_params']['finetune']['split'].get('sessions')
        keep = np.isin(load_sessions(cfg, args, subs, subj), sessions) if sessions is not None else np.ones(len(y), bool)
        T, Np, Cv, D = z.shape
        per = {k: {} for k in ('normal',) + tuple(transforms)}
        changed, flipped = {tf: [] for tf in transforms}, {tf: [] for tf in transforms}
        for s in subs:
            tr, te = keep & (subj != int(s)), keep & (subj == int(s))
            mu, W = _pca_axes(z[torch.from_numpy(tr)].reshape(-1, D), n_pca)
            feat = lambda zz: ((zz - mu) @ W).reshape(T, -1).numpy()
            f = feat(z)
            m, sd = f[tr].mean(0), f[tr].std(0) + 1e-6
            clf = GridSearchCV(RidgeClassifier(class_weight='balanced'), {'alpha': ALPHAS}, cv=LeaveOneGroupOut(),
                               scoring='balanced_accuracy').fit((f[tr] - m) / sd, y[tr], groups=subj[tr])
            p0 = clf.predict((f[te] - m) / sd)
            per['normal'][s] = float(balanced_accuracy_score(y[te], p0))
            hand_te = np.isin(y[te], list(hand.values()))
            for tf in transforms:
                p = clf.predict((feat(zt[tf])[te] - m) / sd)
                per[tf][s] = float(balanced_accuracy_score(y[te], p))
                changed[tf].append(float((p != p0).mean()))
                other = np.where(y[te] == hand['left'], hand['right'], hand['left'])
                flipped[tf].append(float((p[hand_te] == other[hand_te]).mean()))
        res[ds] = {'sessions': sessions, 'hand_classes': hand,
                   'balanced_accuracy': {k: float(np.mean(list(v.values()))) for k, v in per.items()},
                   'per_subject': per,
                   'prediction_changed': {tf: float(np.mean(v)) for tf, v in changed.items()},
                   'hand_predicted_as_other_hand': {tf: float(np.mean(v)) for tf, v in flipped.items()}}
        r = res[ds]
        print(f'  {ds:12s} normal {r["balanced_accuracy"]["normal"] * 100:5.1f} | ' + ' | '.join(
            f'{tf} {r["balanced_accuracy"][tf] * 100:5.1f} (changed {r["prediction_changed"][tf] * 100:.0f}%, '
            f'other hand {r["hand_predicted_as_other_hand"][tf] * 100:.0f}%)' for tf in transforms))
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


def compute(ctx):
    coord_robustness(ctx.config, ctx.checkpoint, ctx.numbers('coord_robustness.json'))
