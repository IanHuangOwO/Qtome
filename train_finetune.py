"""Finetune a FeatureHead on a frozen Qlet backbone.

The backbone never trains: qlet features come from the amplitude cache (model/Qlet/finetune/feature_cache.py), raw
features from the compiled dataset; only the head is optimised (fp32, batches indexed from RAM).
training_params.finetune.split picks a split type (SPLITS: loso, subject_kfold, eval_subjects,
kfold, blocked_kfold, fewshot). Every run writes artifacts/group_eval.json: per-subject tail (mean of the last 10
epochs) and last-epoch balanced accuracy."""
import argparse, json, os, subprocess, sys, time

import torch
import torch.optim as optim

from IO.config import apply_overrides, load_config, model_output_root, setup_logger
from IO.splits import SPLITS, all_subjects, load_sessions, make_runs, resolve_subjects
from model.Qlet.finetune.fit import build_head_factory, run_closed_form, run_one
from model.Qlet.finetune.protocol import apply_protocol
from model.Qlet.finetune.sources import make_source



def env_atom():
    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    try:
        import mne
        mne_version = mne.__version__
    except ImportError:
        mne_version = None
    return {'git_commit': sh('git', 'rev-parse', 'HEAD'), 'git_dirty': bool(sh('git', 'status', '--porcelain')),
            'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'mne': mne_version}


def main():
    ap = argparse.ArgumentParser(description='Finetune a FeatureHead on a frozen Qlet backbone')
    ap.add_argument('--config', default='configs/Qlet/finetune.template.json')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a config value, dotted path, JSON value '
                         '(repeatable), e.g. --set training_params.finetune.learning_rate=0.003')
    args = ap.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    config = apply_overrides(apply_protocol(config), args.set)      # a named protocol, then --set still wins
    tp = config['training_params']['finetune']
    if tp.get('num_threads'):          # CPU threads for this process (parallel runs share the cores)
        torch.set_num_threads(int(tp['num_threads']))
    if 'split' not in tp:
        raise ValueError(f"training_params.finetune.split is required (type: one of {sorted(SPLITS)})")
    # output_path: where this run writes under output/Qlet/ -- separate from model_name (a
    # clean identity string), same split as train_pretrain.py's. Falls back to
    # model_name for configs that don't set it.
    base = os.path.join(model_output_root(config, 'finetune'), tp.get('output_path', tp.get('model_name', 'default_finetune_run')))
    artifact_dir = os.path.join(base, 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    logger, timestamp = setup_logger(artifact_dir)
    snapshot = dict(config, env=env_atom())
    for name in ('config.json', f'config_{timestamp}.json'):
        with open(os.path.join(artifact_dir, name), 'w') as f:
            json.dump(snapshot, f, indent=2)
    dp = config['dataset_params']['finetune']
    if len(dp) != 1:
        raise ValueError("finetune runs one dataset at a time: dataset_params.finetune must have exactly one entry")
    ds_name, ds_args = next(iter(dp.items()))
    pool = resolve_subjects(ds_args['subject_to_use'], all_subjects(ds_args['dataset_path']))
    device = tp.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    source = make_source(config, ds_name, pool, device)
    labels = source.labels.numpy()
    num_classes = int(labels.max()) + 1
    assert set(labels.tolist()) == set(range(num_classes)), f"labels must be contiguous 0..{num_classes - 1}"
    head_cfg, new_head = build_head_factory(config, source, num_classes)
    logger.info(f"dataset={ds_name} pool={len(pool)} subjects, {len(labels)} trials, classes={num_classes}, head={head_cfg}")
    subject_data = source.subject_data.numpy()
    runs = make_runs(tp['split'], pool, subject_data, labels, load_sessions(config, ds_args, pool, subject_data))
    result = {}
    for run in runs:
        tag = f"{ds_name}_{run['name']}"
        dirs = {'ckpt': os.path.join(base, 'finetune', f"run_{run['name']}")}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        groups = (run_closed_form(config, run, source, tag, logger, dirs['ckpt']) if tp.get('fit') == 'closed_form'
                  else run_one(config, run, source, head_cfg, new_head, tag, dirs, logger, device))
        for g, d in groups.items():
            logger.info(f"  [{run['name']}] group {g}: n={d['n_subjects']} mean_tail={d['mean_tail']:.4f} mean_last={d['mean_last']:.4f}")
        result[run['name']] = {'train_subjects': run['train_subjects'], 'epochs': tp['epochs'],
                               'tail_epochs': min(10, tp['epochs']), 'groups': groups}
        with open(os.path.join(artifact_dir, 'group_eval.json'), 'w') as f:      # rewritten per run: partial results survive
            json.dump(result, f, indent=2)
    logger.info("Finetuning complete.")


if __name__ == '__main__':
    main()
