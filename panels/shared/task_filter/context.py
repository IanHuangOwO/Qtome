"""The saved task filter of one source and the backbone it was built on: what every task_filter_* panel starts from."""
import json
import os
from types import SimpleNamespace

import torch

from panels.shared.task_filter.backbone import Backbone
from panels.shared.task_filter.fit import SOURCES, build


def filter_context(ctx, source, rebuild=False):
    """(out dir, test context) of one filter source ('multi' or a dataset) for an analyze.py pretrain run: the saved
    filter <analysis>/filter/<source>/filter.pt, built first if missing (or rebuilt). The backbone is
    output/Qtome/<backbone>, three levels above the checkpoint (<backbone>/pretrain/checkpoint/last.pth)."""
    bb = os.path.dirname(os.path.dirname(os.path.dirname(ctx.checkpoint)))
    if not hasattr(ctx, '_tf_backbone'):
        ctx._tf_backbone = Backbone(bb)
    sources = ctx.args.filter_sources or list(SOURCES)
    out = f'{ctx.numbers_dir}/filter/{source}'; os.makedirs(out, exist_ok=True); path = f'{out}/filter.pt'
    if rebuild or not os.path.exists(path):
        filt = build(bb, source, ctx._tf_backbone, sources)
        torch.save(filt, path); print(f'filter saved: {path}')
    else:
        filt = torch.load(path, weights_only=False)
        assert source != 'multi' or filt['sources'] == sources, f'{path} was built from {filt["sources"]}: run task_filter_build'
    return out, SimpleNamespace(bb=bb, name=os.path.basename(bb), backbone=ctx._tf_backbone, filt=filt)


def write_table(stem, rows, lines):
    """<stem>.json (rows) and <stem>.log (the printed tables, without the progress lines)."""
    json.dump(rows, open(stem + '.json', 'w'), indent=1)
    open(stem + '.log', 'w').write('\n'.join(l for l in lines if not l.endswith(' done')) + '\n')
