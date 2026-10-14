"""Task filter: a reconstruction that keeps only what classification heads use (model/Qtome/Qtome_filter_tools.py), its tests
on datasets it never saw (tools/analysis/task_filter_eval.py) and its panel (tools/viz/task_filter_panel.py).

    python task_filter.py --run <backbone> --source multi build                  # fit and save the filter
    python task_filter.py --run <backbone> --source multi eval mi [--target BNCI2014001 BNCI2014004]
    python task_filter.py --run <backbone> --source multi eval p300|noise [--target ...]
    python task_filter.py --run <backbone> --source multi panel --target BNCI2014004 [--share 0.1]
    python task_filter.py --run <backbone> compare --target BNCI2014008 --single BNCI2014009   # vs ASR and ICLabel

--source: 'multi' (the multi-task filter from the MI, P300 and SSVEP heads; --sources changes them) or one dataset (the
single-task filter from its own heads). Either needs the backbone's loso cells on imputed 64-channel gains, finetune/allatom/<ds>_impute64_loso_seed1
(dataset_params.finetune.<ds>.impute_missing_channels=true). eval and panel build the filter first if it is not saved.
Output: output/Qtome/<backbone>/task_filter/<source>/ -- filter.pt, eval_<test>_<target>.json and .log (the tables),
panel_<target>_<share>.png. compare: the single-task filter of --single and the multi-task filter on one target with
injected artifacts, against ASR and ICLabel -> task_filter/compare/compare_<target>.json, .log, .png."""
import argparse
import json
import os
from types import SimpleNamespace

import torch

from model.Qtome.Qtome_filter_tools import SHARES, SOURCES, Backbone, build
from tools.analysis import QTOME_OUTPUT
from tools.analysis.task_filter_eval import DEFAULT_TARGETS, TESTS, compare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='backbone name: output/Qtome/<run>')
    ap.add_argument('--source', default='multi', help="'multi' or a dataset name")
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), help='multi: the datasets whose heads build it')
    ap.add_argument('--threads', type=int, default=4)
    sub = ap.add_subparsers(dest='step', required=True)
    sub.add_parser('build', help='fit the filter and save it (again)')
    ev = sub.add_parser('eval', help='test the filter on unseen datasets')
    ev.add_argument('test', choices=sorted(TESTS))
    ev.add_argument('--target', nargs='+', help=f'datasets (default {DEFAULT_TARGETS})')
    pa = sub.add_parser('panel', help='draw the panel of the filter on unseen datasets')
    pa.add_argument('--target', nargs='+', required=True)
    pa.add_argument('--share', type=float, default=0.1, choices=SHARES)
    co = sub.add_parser('compare', help='single-task and multi-task filters against ASR and ICLabel, injected artifacts')
    co.add_argument('--target', nargs='+', required=True)
    co.add_argument('--single', required=True, help='source dataset of the single-task filter (not a target)')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    bb = f'{QTOME_OUTPUT}/{args.run}'; backbone = Backbone(bb)

    def load(source, rebuild=False):                                      # the saved filter of a source, built first if needed
        out = f'{bb}/task_filter/{source}'; os.makedirs(out, exist_ok=True); path = f'{out}/filter.pt'
        if rebuild or not os.path.exists(path):
            filt = build(bb, source, backbone, args.sources)
            torch.save(filt, path); print(f'filter saved: {path}')
        else:
            filt = torch.load(path, weights_only=False)
            assert source != 'multi' or filt['sources'] == args.sources, f'{path} was built from {filt["sources"]}: run build'
        return out, SimpleNamespace(bb=bb, name=args.run, backbone=backbone, filt=filt)

    if args.step == 'compare':
        from tools.viz.task_filter_compare import draw as draw_compare
        ctxs = {'single-task': load(args.single)[1], 'multi-task': load('multi')[1]}
        out = f'{bb}/task_filter/compare'; os.makedirs(out, exist_ok=True)
        for dst in args.target:
            assert dst != args.single and dst not in args.sources, f'{dst} helped build a filter: not an unseen target'
            lines = []
            log = lambda s: (print(s, flush=True), lines.append(s))
            rows = compare(ctxs, dst, log=log); stem = f'{out}/compare_{dst}'
            json.dump(rows, open(stem + '.json', 'w'), indent=1)
            open(stem + '.log', 'w').write('\n'.join(l for l in lines if not l.endswith(' done')) + '\n')
            draw_compare(rows, dst, {'single-task': args.single, 'multi-task': ' + '.join(args.sources)}, stem + '.png')
        return
    out, ctx = load(args.source, args.step == 'build')
    if args.step == 'eval':
        for dst in args.target or DEFAULT_TARGETS[args.test]:
            lines = []
            log = lambda s: (print(s, flush=True), lines.append(s))
            rows = TESTS[args.test](ctx, dst, log=log)
            stem = f'{out}/eval_{args.test}_{dst}'
            json.dump(rows, open(stem + '.json', 'w'), indent=1)
            open(stem + '.log', 'w').write('\n'.join(l for l in lines if not l.endswith(' done')) + '\n')
    elif args.step == 'panel':
        from tools.viz.task_filter_panel import draw
        for dst in args.target:
            draw(ctx, dst, args.share, f'{out}/panel_{dst}_{args.share:g}.png')


if __name__ == '__main__':
    main()
