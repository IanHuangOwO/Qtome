"""Pretrain-stage analysis of one backbone: runs a preset (or --panel list) of tools/panels/ panels.

    python analysis_pretrain.py --run <backbone>            # output/<backbone>/pretrain/checkpoint/last.pth
    python analysis_pretrain.py --checkpoint <path.pth> [--preset quick] [--panel snapshot ...]
    python analysis_pretrain.py --panel profile [--train]   # no checkpoint: the pretrain template

The config is the run's own artifacts/config.json (next to the checkpoint), deep-merged with an optional
--config overlay; the model is rebuilt from the checkpoint's build_config. Output:
output/<output_path>/analysis/ (e.g. output/<backbone>/pretrain/analysis/: numbers, logs, markdown) and every figure in
output/<backbone>/analysis/pretrain/. Presets: tools/panels PRESETS.

Task filter (panels task_filter_build / _filter / _eval / _panel / _compare, on request; the filter and the tables ->
analysis/task_filter/<source>/ and .../compare/, its figures -> analysis/pretrain/task_filter/):
    python analysis_pretrain.py --run <backbone> --panel task_filter_eval [--filter-source multi] [--filter-test mi p300 noise] [--target <ds> ...]
    python analysis_pretrain.py --run <backbone> --panel task_filter_panel --target BNCI2014004 [--share 0.1]
    python analysis_pretrain.py --run <backbone> --panel task_filter_compare --target BNCI2014008 [--single BNCI2014009]
"""
import argparse
import json
import os
import sys

import torch

from tools.analysis import _deep_merge, resolve_output_dir
from tools.panels import PRESETS, PanelContext, discover_panel_names, resolve_panels, run_panels
from tools.analysis import QTOME_OUTPUT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--run', help='backbone name: output/<run>/pretrain/checkpoint/last.pth')
    src.add_argument('--checkpoint', help='a pretrain checkpoint .pth')
    ap.add_argument('--config', help='JSON overlay deep-merged onto the run config')
    ap.add_argument('--preset', default='standard', choices=sorted(PRESETS['pretrain']))
    ap.add_argument('--panel', action='append', default=[],
                    help=f'run these panels instead of the preset: {discover_panel_names("pretrain")}')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--datasets', nargs='+', help='snapshot/atom_distribution/codebook: these pretrain datasets')
    ap.add_argument('--max-trials', type=int, default=1000, help='codebook/atom_distribution: trials per dataset')
    ap.add_argument('--max-windows', type=int, default=512, help='backbone_eval: held-out windows')
    ap.add_argument('--dup-threshold', type=float, default=0.9, help='atom_duplicates: pair threshold')
    ap.add_argument('--train', action='store_true', help='profile: train-mode timing')
    ap.add_argument('--threads', type=int, help='torch CPU threads (default: torch\'s)')
    ap.add_argument('--filter-source', default='multi', help="task_filter_*: 'multi' or a dataset name")
    ap.add_argument('--filter-sources', nargs='+', help='task_filter_*: the datasets whose heads build the multi filter (default: Qtome_filter_tools SOURCES)')
    ap.add_argument('--filter-test', nargs='+', choices=['mi', 'p300', 'noise'], help='task_filter_eval: tests (default all)')
    ap.add_argument('--target', nargs='+', help='task_filter_*: unseen datasets (eval: default per test)')
    ap.add_argument('--share', type=float, default=0.1, choices=[0.5, 0.25, 0.1], help='task_filter_panel: share of cells kept')
    ap.add_argument('--single', help='task_filter_compare: optional, source dataset of a single-task filter to add (not a target); default: the multi-task filter only')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    names = resolve_panels('pretrain', args.preset, args.panel)
    checkpoint = args.checkpoint or (f'{QTOME_OUTPUT}/{args.run}/pretrain/checkpoint/last.pth' if args.run else None)
    if checkpoint:
        config = json.load(open(os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'artifacts', 'config.json')))
    elif names == ['profile']:
        config = json.load(open('configs/Qtome/pretrain.template.json'))
    else:
        ap.error('--run or --checkpoint is required (only the profile panel runs without one)')
    if args.config:
        config = _deep_merge(config, json.load(open(args.config)))
    out_dir = resolve_output_dir(config, 'analysis') if checkpoint else 'output/tools-profile'
    os.makedirs(out_dir, exist_ok=True)
    # numbers in <backbone>/pretrain/analysis, figures in <backbone>/analysis/pretrain (the same two levels above out_dir)
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), 'analysis', 'pretrain') if checkpoint else out_dir
    os.makedirs(fig_dir, exist_ok=True)

    ctx = PanelContext(stage='pretrain', config=config, out_dir=out_dir, fig_dir=fig_dir, args=args,
                       device=torch.device(args.device), checkpoint=checkpoint)
    sys.exit(1 if run_panels(names, ctx) else 0)


if __name__ == '__main__':
    main()
