"""
Builds dataset_params.pretrain for the pretrain corpus (design 2026-09-30, event-free caches 2026-10-05, hierarchical
smoothed balance 2026-10-21): as many subjects and recording configurations as possible, paradigms balanced softly.

Three levels, each with one smoothing exponent (temperature sampling):
  1. paradigm group: hours = TOTAL_HOURS x the mean of the fixed GROUPS target and sqrt(unique subjects) normalised;
  2. unit (a dataset, or a cohort whose members are tasks of the same people / releases of one study): hours inside a group
     proportional to unique subjects ** ALPHA, no more than the unit has;
  3. subject: equal time inside a dataset, the dataset's water level t seconds (every subject contributes min(own, t)):
     written as dataset_params.pretrain.<ds>.max_windows_per_subject, the first windows of one seeded permutation
     (IO/dataset.py), so a smaller corpus size is a subset of a larger one.
A cohort of the same people (PURSUE, ErpCore2021, GraspAndLift) counts its people once and splits its hours equally over
its tasks; a cohort of different people (BETA, Inria) splits them in proportion to subjects. Subjects are never dropped
except by the MAX_SUBJECTS cap (a fixed seeded pick, the same for every corpus size). Hours are seconds of real signal (valid
samples) in the _cont5 caches. The corpus sizes multiply in on top: a run sets preprocess_params.window_fraction (tiny 0.05 /
small 0.2 / medium 0.5; the template's 1.0 = large), one window per subject is the floor at every size. One template, one
corpus list for every size.

    python -m tools.misc.build_pretrain_corpus --out configs/Qtome/pretrain.template.json

--out must already exist (a pretrain config); only its dataset_params.pretrain is replaced.
"""
import argparse
import glob
import json
import os
import random

import numpy as np

TOTAL_HOURS = 210.0
# group -> (target share, units). A unit is a dataset name or (cohort, same_people, [members]). Left out 2026-10-05:
# Stieger2021, Yang2025, LEMON, Nieuwland2018_N400, GuttmannFlury2025_* (raw > 100 GB, disk), Liu2024 (added after the build);
# in-house sets dropped 2026-10-21.
GROUPS = {
    'motor': (0.25, ['Lee2019_MI', 'Dreyer2023', 'Cho2017', 'Weibo2014', 'Schirrmeister2017', 'Chang2025',
                     ('GraspAndLift', True, ['GraspAndLift_Train', 'GraspAndLift_Test'])]),
    'SSVEP': (0.25, ['Lee2019_SSVEP', 'Wang2016', 'Liu2022EldBETA', 'Kim2025BetaRange',
                     ('BETA', False, ['BETA_4s', 'BETA_3s'])]),
    'ERP': (0.25, ['ERP_Longitudinal', 'BI2014a', 'BI2014b', 'BI2015a', 'BI2015b',
                   ('Inria', False, ['Inria_Train', 'Inria_Test']),
                   ('ErpCore2021', True, [f'ErpCore2021_{t}' for t in ('ERN', 'MMN', 'N170', 'N2pc', 'N400', 'P3')]),
                   ('PURSUE', True, [f'PURSUE_{t}' for t in ('LRP_ERN', 'MMN', 'N170', 'N2pc', 'N400', 'P300')])]),
    'rest / clinical': (0.125, ['SRM_RestingState', 'MDD_Mumtaz', 'Neonatal_Helsinki', 'UCSD_PD', 'SPIS',
                                'NMT_Clinical', 'LSE_IndiaTanzania', 'PD_Rest', 'Rest_PrePostCognitive']),
    'cognitive / affective': (0.125, ['BCMI_MusicEmotion', 'BCIC2020-3', 'STEW', 'Gao2026']),
}
SUFFIX = 'fs200_bp0.5-100.0_cont5'
WINDOW_S = 5                    # seconds per cached window (cont5)
ALPHA = 0.75                    # unit hours ~ unique subjects ** ALPHA (0 = equal per unit, 1 = equal per subject)
GROUP_BLEND = 0.5               # group share = (1 - GROUP_BLEND) x target + GROUP_BLEND x sqrt(unique subjects) share
# Datasets with more subjects keep a fixed seeded pick of this many: one window per subject is the floor at every corpus size,
# so NMT_Clinical (2417) / LSE_IndiaTanzania (2000) / Rest_PrePostCognitive (608) would fill the tiny corpus (5%) alone.
MAX_SUBJECTS = 1000


def subject_hours(ds):
    """{subject: hours of real signal}, the seeded MAX_SUBJECTS pick for a larger dataset."""
    out = {}
    for f in glob.glob(f'datas/pretrain/{ds}/cache/*_{SUFFIX}.npz'):
        with np.load(f) as z:
            out[os.path.basename(f).split(f'_{SUFFIX}')[0]] = (z['valid_end'] - z['valid_start']).sum() / 200 / 3600
    if not out:
        raise SystemExit(f'{ds}: no compiled {SUFFIX} cache')
    if len(out) > MAX_SUBJECTS:
        keep = random.Random(f'subjects/{ds}').sample(sorted(out), MAX_SUBJECTS)
        out = {s: out[s] for s in keep}
    return out


def split_budget(budget, weights, avail):
    """Shares of budget proportional to weights, each at most its avail; the excess of a clipped one goes to the others.
    {key: weight}, {key: avail} -> {key: hours} (sum = min(budget, sum(avail)))."""
    out, free = {}, set(weights)
    while free:
        z = sum(weights[k] for k in free)
        over = [k for k in free if budget * weights[k] / z > avail[k]]
        if not over:
            out.update({k: budget * weights[k] / z for k in free})
            break
        for k in over:
            out[k] = avail[k]
            budget -= avail[k]
            free.discard(k)
    return out


def level_seconds(hours_by_subject, alloc_h):
    """Water level t (seconds) with sum(min(own, t)) = alloc_h over the subjects; None when alloc_h covers all of them."""
    own = sorted(h * 3600 for h in hours_by_subject.values())
    if alloc_h * 3600 >= sum(own) - 1e-9:
        return None
    left, n = alloc_h * 3600, len(own)
    for i, h in enumerate(own):
        if h * (n - i) >= left:
            return left / (n - i)
        left -= h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.out))
    dp, total = {}, 0.0
    # every unit: members, same_people flag, per-member subjects; unique subjects of the unit
    groups = {}
    for group, (target, units) in GROUPS.items():
        glist = []
        for u in units:
            name, same, members = (u, True, [u]) if isinstance(u, str) else u
            subj = {ds: subject_hours(ds) for ds in members}
            n_each = [len(s) for s in subj.values()]
            glist.append({'name': name, 'same': same, 'members': members, 'subj': subj,
                          'n': max(n_each) if same else sum(n_each),
                          'avail': {ds: sum(s.values()) for ds, s in subj.items()}})
        groups[group] = (target, glist)
    sq = {g: sum(u['n'] for u in gl) ** 0.5 for g, (_, gl) in groups.items()}
    for group, (target, glist) in groups.items():
        share = (1 - GROUP_BLEND) * target + GROUP_BLEND * sq[group] / sum(sq.values())
        budget = share * TOTAL_HOURS
        unit_h = split_budget(budget, {u['name']: u['n'] ** ALPHA for u in glist},
                              {u['name']: sum(u['avail'].values()) for u in glist})
        used = 0.0
        for u in glist:
            h_u = unit_h[u['name']]
            # members: equal hours for the tasks of the same people, in proportion to subjects for different people
            wts = {ds: (1.0 if u['same'] else len(u['subj'][ds])) for ds in u['members']}
            member_h = split_budget(h_u, wts, u['avail'])
            for ds in u['members']:
                subj = u['subj'][ds]
                t = level_seconds(subj, member_h[ds])
                dp[ds] = {'dataset_path': f'datas/pretrain/{ds}', 'subject_to_use': sorted(subj, key=lambda s: (len(s), s)),
                          'channels_to_use': ['all']}
                if t is None:
                    got = sum(subj.values())
                else:
                    max_w = max(1, int(round(t / WINDOW_S)))
                    dp[ds]['max_windows_per_subject'] = max_w
                    got = sum(min(h * 3600, max_w * WINDOW_S) for h in subj.values()) / 3600
                used += got
                print(f'  {group:22} {ds:24} {len(subj):5} subjects {sum(subj.values()):7.1f} h -> {got:5.1f} h '
                      f'({got * 3600 / len(subj):4.0f} s per subject)'
                      + (f' max_windows_per_subject {dp[ds]["max_windows_per_subject"]}' if t is not None else ''))
        print(f'{group}: {used:.1f} h (share {share:.1%} of {TOTAL_HOURS:.0f} h, target {target:.1%})')
        total += used
    print(f'total {total:.1f} h')
    cfg['dataset_params']['pretrain'] = dp
    with open(args.out, 'w') as f:
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
