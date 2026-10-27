"""class_snapshots: for one finetune head (--checkpoint .../<cell>/finetune/run_<fold>/head.pth), per class one correctly and one wrongly
classified trial, each rendered as the recon grid and qlet gallery -> class_snapshots/<cell>_<fold>/. The config is the run's
artifacts/config.json; trials come from the fold's held-out subjects (group_eval.json), so for a within-subject split they include its
train trials."""
import copy
import glob
import json
import os

import numpy as np

from IO.dataset import build_dataset_from_config
from panels.core.context import load_model
from panels.core.numbers import load_npz, save_npz
from panels.shared.snapshot import (build_finetune_bundle, draw_gallery, draw_recon, load_snapshot, predict_all,
                                    snapshot_numbers)

PRESETS = set()


def add_args(flags):
    flags.add('--checkpoint', help='class_snapshots: a head.pth')


def _tag(ctx):
    ckpt = ctx.args.checkpoint
    if not ckpt:
        raise RuntimeError('class_snapshots needs --checkpoint <.../finetune/run_<fold>/head.pth>')
    fold = os.path.basename(os.path.dirname(ckpt))[len('run_'):]
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt)))
    return ckpt, fold, run_dir, f'{os.path.basename(run_dir)}_{fold}'


def compute(ctx):
    ckpt, fold, run_dir, tag = _tag(ctx)
    cfg = json.load(open(os.path.join(run_dir, 'artifacts', 'config.json')))
    ge = json.load(open(os.path.join(run_dir, 'artifacts', 'group_eval.json')))
    subjects = sorted({s for g in ge[fold]['groups'].values() for s in g['subjects']})
    name, ds_args = next(iter(cfg['dataset_params']['finetune'].items()))
    cfg = copy.deepcopy(cfg)
    cfg['dataset_params']['finetune'] = {name: {**ds_args, 'subject_to_use': subjects}}
    ds = build_dataset_from_config(cfg, mode='finetune')
    model = load_model(cfg, ckpt, ctx.device, mode='finetune')

    pp = cfg['preprocess_params']
    preds, labels = predict_all(model, ds, pp['patch_length'], pp.get('patch_stride', pp['patch_length']), ctx.device)
    meta = json.load(open(os.path.join(ds_args['dataset_path'], 'metadata.json')))
    names = meta.get('data_metadata', {}).get('targets', {})
    for old in glob.glob(ctx.numbers('class_snapshots', tag, '*.npz')):
        os.remove(old)
    print(f"  {name} {fold}: subjects {subjects}, accuracy {np.mean(preds == labels):.3f} on {len(labels)} trials")
    for c in range(int(labels.max()) + 1):
        cname = ''.join(ch if ch.isalnum() else '_' for ch in names.get(str(c), {}).get('label', f'class{c}'))
        for status, hit in (('correct', preds == c), ('wrong', preds != c)):
            idx = np.flatnonzero((labels == c) & hit)
            if not len(idx):
                print(f"  class {c} ({cname}): no {status} trial")
                continue
            i = int(idx[0])
            bundle, _ = build_finetune_bundle(model, ds, i, cfg, ctx.device,
                                              subject_id=int(ds.base_dataset.subject_data[i]),
                                              tag=f'_class{c}_{cname}_{status}')
            save_npz(ctx.numbers('class_snapshots', tag, f'class{c}_{cname}_{status}.npz'), **snapshot_numbers(bundle, cfg))


def draw(ctx):
    tag = _tag(ctx)[3]
    out_dir = ctx.figure('class_snapshots', tag, '')
    for path in sorted(glob.glob(ctx.numbers('class_snapshots', tag, '*.npz'))):
        snap = load_snapshot(load_npz(path))
        draw_recon(snap, snap.cfg, out_dir)
        draw_gallery(snap, snap.cfg, out_dir, cmap=ctx.cmap)
