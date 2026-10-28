"""The context every panel receives: the run's config and paths, the lazily built model, the device and the parsed options.
Numbers go under ctx.numbers_dir, figures under ctx.fig_dir -- in the docs vault, not in output/ (pretrain: numbers in
output/Qlet/<backbone>/pretrain/analysis, figures in docs/figures/runs/<backbone>/pretrain; group: numbers in --out, figures in
docs/figures/<name of --out>)."""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from IO.config import QLET_OUTPUT, resolve_output_dir

FIGURES = 'docs/figures'          # every figure the panels draw; the Obsidian vault embeds them from here


def deep_merge(base: dict, override: dict) -> dict:
    """The panels' --config option: a partial config merged over the run's own config."""
    result = dict(base)
    for k, v in override.items():
        result[k] = deep_merge(result[k], v) if isinstance(result.get(k), dict) and isinstance(v, dict) else v
    return result


def load_model(config, checkpoint, device, mode='pretrain'):
    """Pretrain: the backbone rebuilt from the checkpoint's own build_config (random weights from config if the file is missing), eval.
    mode='finetune' rebuilds the full FinetuneModel (frozen backbone + head) from the head checkpoint's own head_config and
    backbone_checkpoint path (no shape inference; a missing file raises)."""
    if mode == 'finetune':
        from model.factory import load_finetune_checkpoint
        return load_finetune_checkpoint(config, checkpoint, device)
    from model.factory import build_from_checkpoint, build_pretrain_from_config
    if checkpoint and os.path.exists(checkpoint):
        model = build_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False)).to(device)
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint!r}, using random weights.")
        model = build_pretrain_from_config(config, mode=mode).to(device)
    model.eval()
    return model


@dataclass
class Ctx:
    stage: str
    args: object                       # argparse.Namespace: the common options plus every selected panel's own
    numbers_dir: str
    fig_dir: str
    config: dict = field(default_factory=dict)
    device: object = None
    checkpoint: Optional[str] = None   # pretrain
    groups: dict = field(default_factory=dict)   # group: {name: backbone}
    ref: Optional[str] = None
    head: Optional[str] = None
    redraw: bool = False
    _model: object = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.config, self.checkpoint, self.device, mode='pretrain')
        return self._model

    @property
    def cmap(self):
        return self.config.get('training_params', {}).get('visualize_params', {}).get('cmap', 'YlOrRd')

    def numbers(self, *parts):
        """A path under numbers_dir; its folder is created."""
        path = os.path.join(self.numbers_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def figure_in(self, stage, *parts):
        """A path under the sibling figure folder <analysis>/<stage>/ (e.g. 'filter'); its folder is created."""
        path = os.path.join(os.path.dirname(self.fig_dir), stage, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def figure(self, *parts):
        """A path under fig_dir; its folder is created."""
        path = os.path.join(self.fig_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


def add_stage_flags(p, stage):
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    if stage == 'pretrain':
        src = p.add_mutually_exclusive_group()
        src.add_argument('--run', help='backbone name: output/Qlet/<run>/pretrain/checkpoint/last.pth')
        src.add_argument('--checkpoint', help='a pretrain checkpoint .pth')
        p.add_argument('--config', help='JSON overlay deep-merged onto the run config')
        p.add_argument('--threads', type=int, help="torch CPU threads (default: torch's)")
    else:
        p.add_argument('--group', action='append', required=True, metavar='NAME=BACKBONE',
                       help='a backbone to compare (repeatable, report order)')
        p.add_argument('--ref', help='the reference group (tests against it)')
        p.add_argument('--head', default='frozen_learned', help='finetune label: output/Qlet/<backbone>/finetune/<head>/')
        p.add_argument('--out', help='output dir (default output/Qlet/<common name of the backbones>_analysis/)')


def default_analysis_dir(groups):
    """Where the analysis of several backbones goes, at the level of the backbones: output/Qlet/<name>_analysis, name = what the backbones'
    names have in common (qlet_small_v4flat_onephase_s1/s2/s3 -> qlet_small_v4flat_onephase), a lone backbone's own name, else the group names."""
    bbs = sorted(set(groups.values()))
    parts = [b.split('_') for b in bbs]
    k = next((i for i, t in enumerate(zip(*parts)) if len(set(t)) > 1), min(map(len, parts)))   # leading name parts all share (whole parts: _s1 / _s2 differ)
    name = bbs[0] if len(bbs) == 1 else '_'.join(parts[0][:k]) if k >= 2 else '_'.join(groups)
    return os.path.join(QLET_OUTPUT, f'{name}_analysis')


def make_ctx(stage, args, panel_modules, error):
    """panel_modules: the selected panel modules (for NEEDS_CHECKPOINT). error: argparse's parser.error."""
    device = torch.device(args.device)
    if stage == 'group':
        groups = dict(g.split('=', 1) for g in args.group)
        if args.ref and args.ref not in groups:
            error(f'--ref {args.ref!r} is not a group name ({list(groups)})')
        out = args.out or default_analysis_dir(groups)
        os.makedirs(out, exist_ok=True)
        fig_out = os.path.join(FIGURES, os.path.basename(os.path.normpath(out)))
        os.makedirs(fig_out, exist_ok=True)
        return Ctx(stage, args, out, fig_out, device=device, groups=groups, ref=args.ref, head=args.head, redraw=args.redraw)

    if args.threads:
        torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint or (f'{QLET_OUTPUT}/{args.run}/pretrain/checkpoint/last.pth' if args.run else None)
    if checkpoint:
        config = json.load(open(os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'artifacts', 'config.json')))
    elif all(not getattr(m, 'NEEDS_CHECKPOINT', True) for m in panel_modules):
        config = json.load(open('configs/Qlet/pretrain.template.json'))
    else:
        error('--run or --checkpoint is required (only panels with NEEDS_CHECKPOINT = False run without one)')
    if args.config:
        config = deep_merge(config, json.load(open(args.config)))
    if checkpoint:
        numbers_dir = resolve_output_dir(config, 'analysis')
        # numbers in <backbone>/pretrain/analysis, figures in docs/figures/runs/<backbone>/pretrain
        fig_dir = os.path.join(FIGURES, 'runs', os.path.basename(os.path.dirname(os.path.dirname(numbers_dir))), 'pretrain')
    else:
        numbers_dir = fig_dir = 'output/tools-profile'
    os.makedirs(numbers_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return Ctx(stage, args, numbers_dir, fig_dir, config=config, device=device, checkpoint=checkpoint, redraw=args.redraw)
