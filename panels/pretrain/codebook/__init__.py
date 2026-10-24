"""codebook: cross-dataset Q-atom-usage diagnostics -> analysis/pretrain/codebook/. One dataset per pretrain entry (usage stays
attributable to its source), real trials with their labels, each capped at --max-trials trials (shared/trials.cap_subjects).
Figures: embedding scatters (per patch and per trial, by dataset and by target), dataset relation, usage by dataset, template
similarity, Q-atom phase / topography / identity consistency, patch-position consistency and event dynamics per dataset.
Options: --datasets, --max-trials."""
import copy
import glob
import os
import random

import numpy as np

from IO.dataset import build_dataset_from_config
from IO.event_onset import event_onset_patch
from panels.core.numbers import load_npz, save_npz
from panels.pretrain.codebook import plots
from panels.pretrain.codebook import readout
from panels.shared.trials import cap_subjects

PRESETS = {'standard'}
UNIT = 'Atom'


def add_args(flags):
    flags.add('--datasets', nargs='+', help='these pretrain datasets (default: every one)')
    flags.add('--max-trials', type=int, default=1000, help='trials per dataset')


def compute(ctx):
    for old in glob.glob(ctx.numbers('codebook', '*.npz')):
        os.remove(old)
    ds_params = ctx.config['dataset_params']['pretrain']
    max_trials, rng = ctx.args.max_trials, random.Random(0)
    by_name = {}
    for name in ctx.args.datasets or list(ds_params):
        cfg = copy.deepcopy(ctx.config)
        subjects, total = cap_subjects(cfg, ds_params[name], max_trials, rng, with_total=True)
        # every dataset stays in memory until the checker ran: load 1.5 x max_trials spread over the subjects, not their
        # every trial (20 subjects of each dataset in full reached > 50 GB, 2026-10-14)
        cfg['dataset_params']['pretrain'] = {name: {**ds_params[name], 'subject_to_use': subjects,
                                                    'trial_fraction': min(1.0, 1.5 * max_trials / max(total, 1))}}
        by_name[name] = build_dataset_from_config(cfg, mode='pretrain', assemble_trials=False)
        print(f"  {name}: {len(subjects)} subject(s)")
    model, device, seed = ctx.model, next(ctx.model.parameters()).device, 0

    usage, labels, t_usage, t_labels, records = readout.sample_usage(model, by_name, max_trials, seed)
    arrays = {'fingerprint': readout.decoder_fingerprint_matrix(model),
              'datasets': np.array(list(by_name))}
    for i, ds in enumerate(by_name):
        trials = [t for t in records if t['dataset'] == ds]
        arrays.update({f'usage_{i}': usage[ds], f'labels_{i}': labels[ds], f'trial_usage_{i}': t_usage[ds],
                       f'trial_labels_{i}': t_labels[ds],
                       f'rec_usage_{i}': np.concatenate([t['usage'] for t in trials]),
                       f'rec_len_{i}': np.array([len(t['usage']) for t in trials]),
                       f'rec_subject_{i}': np.array([t['subject'] for t in trials])})
    save_npz(ctx.numbers('codebook', 'usage.npz'), **arrays)

    ident = readout.identity_numbers(records, model, device, seed)
    arrays = {'circ_var': ident['circ_var'], 'fire_count': ident['fire_count'], 'has_identity': np.array(ident['mats'] is not None)}
    if ident['mats'] is not None:
        arrays.update(within=ident['within'], between=ident['between'], ids=np.array(ident['ids']),
                      per_atom=np.array(ident['per_atom']), mat_datasets=np.array(list(ident['mats'])))
        for j, (ds, (mat, sids)) in enumerate(ident['mats'].items()):
            arrays.update({f'mat_{j}': mat, f'mat_ids_{j}': np.array(sids)})
    save_npz(ctx.numbers('codebook', 'consistency.npz'), **arrays)

    for i, ds in enumerate(by_name):
        ev = readout.event_dynamics_numbers([t for t in records if t['dataset'] == ds], ds, model, device, seed, ctx.config)
        if ev is not None:
            ev = dict(ev, onset_sec=np.array(np.nan if ev['onset_sec'] is None else ev['onset_sec']))
            save_npz(ctx.numbers('codebook', f'event_dynamics_{ds}.npz'), **ev)


def draw(ctx):
    out = lambda name: ctx.figure('codebook', name)
    z = load_npz(ctx.numbers('codebook', 'usage.npz'))
    names = [str(d) for d in z['datasets']]
    usage = {d: z[f'usage_{i}'] for i, d in enumerate(names)}
    labels = {d: z[f'labels_{i}'] for i, d in enumerate(names)}
    t_usage = {d: z[f'trial_usage_{i}'] for i, d in enumerate(names)}
    t_labels = {d: z[f'trial_labels_{i}'] for i, d in enumerate(names)}
    seed, max_points = 0, 3000

    plots.plot_fingerprint_similarity(out('atom_fingerprint_similarity.png'), z['fingerprint'], unit_label=UNIT)
    plots.plot_embedding_scatter_by_dataset(out('patch_embedding_scatter_by_dataset.png'), usage, unit_label=UNIT,
                                            max_points=max_points, random_state=seed)
    plots.plot_embedding_scatter_by_target(out('patch_embedding_scatter_by_target.png'),
                                           out('patch_embedding_scatter_per_dataset.png'), usage, labels,
                                           unit_label=UNIT, max_points=max_points, random_state=seed)
    plots.plot_embedding_scatter_by_dataset(out('trial_embedding_scatter_by_dataset.png'), t_usage, unit_label=UNIT,
                                            max_points=max_points, random_state=seed)
    plots.plot_embedding_scatter_by_target(out('trial_embedding_scatter_by_target.png'),
                                           out('trial_embedding_scatter_per_dataset.png'), t_usage, t_labels,
                                           unit_label=UNIT, max_points=max_points, random_state=seed)

    c = load_npz(ctx.numbers('codebook', 'consistency.npz'))
    plots.plot_atom_phase_consistency(out('atom_phase_consistency.png'), c['circ_var'], c['fire_count'], unit_label=UNIT)
    if bool(c['has_identity']):
        mats = {str(ds): (c[f'mat_{j}'], c[f'mat_ids_{j}'].tolist()) for j, ds in enumerate(c['mat_datasets'])}
        plots.plot_topography_distance(out('atom_topography_distance.png'), mats, unit_label=UNIT)
        plots.plot_atom_identity_consistency(out('atom_identity_consistency.png'), c['within'], c['between'],
                                             c['ids'].tolist(), c['per_atom'].tolist(), unit_label=UNIT)
    plots.plot_dataset_relation(out('dataset_relation.png'), usage, unit_label=UNIT)

    # Unit x Dataset specialization: mean per-unit strength by dataset.
    strength = np.concatenate([usage[d] for d in names], axis=0)                                      # [M_total, Q]
    combined = np.concatenate([np.full(usage[d].shape[0], d) for d in names])                         # [M_total]
    plots.plot_usage_and_activity(out('unit_activity_by_dataset.png'), strength, combined, category_order=names, unit_label=UNIT)

    # Cross-trial, patch-position-aligned consistency (per dataset -- patch position only means the same timeline slot within one
    # dataset's own trial length/patch_len), and the event-locked dynamics.
    for i, ds in enumerate(names):
        lens = np.cumsum(z[f'rec_len_{i}'])[:-1]
        trials = [dict(usage=u, dataset=ds, subject=int(s))
                  for u, s in zip(np.split(z[f'rec_usage_{i}'], lens), z[f'rec_subject_{i}'])]
        plots.plot_patch_position_consistency(out(f'patch_position_consistency_{ds}.png'), trials, unit_label=UNIT, seed=seed,
                                              event_patch=event_onset_patch(ctx.config, ds) if ctx.config else None)
        path = ctx.numbers('codebook', f'event_dynamics_{ds}.npz')
        if os.path.exists(path):
            e = load_npz(path)
            onset = None if np.isnan(e['onset_sec']) else float(e['onset_sec'])
            plots.plot_event_atom_dynamics(out(f'event_atom_dynamics_{ds}.png'), e['t_axis'], e['am'], e['pm'], e['ps'],
                                           onset_sec=onset, unit_label=UNIT,
                                           title_suffix=f' — {ds} ({int(e["n_trials"])} trials x {int(e["n_off"])} offsets, '
                                                        f'step {int(e["fine"])} samp)')
    print(f"  [codebook] -> {ctx.figure('codebook', '')}")
