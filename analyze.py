"""Runs analysis panels (panels/<stage>/<name>.py): each computes its numbers, then draws its figures from them.

    python analyze.py list                                              # every panel, its presets and description
    python analyze.py check                                             # every panel imports and its flags are consistent (no data)
    python analyze.py pretrain --run <backbone> [--preset quick] [--panel snapshot ...] [panel options]
    python analyze.py pretrain --panel profile [--train]                # no checkpoint: the pretrain template
    python analyze.py group --group base=<bb> --group X=<bb> --ref base [--head frozen_learned] [--panel summary ...]   # several backbones (seeds / recipes); `finetune` is the old name
    add --redraw to draw again from the saved numbers (no recompute)

`--panel <name> --help` shows that panel's own options. Pretrain: numbers in output/Qlet/<backbone>/pretrain/analysis/, figures in
output/Qlet/<backbone>/analysis/pretrain/. Group: --out, default output/Qlet/<common name of the backbones>_analysis/. Writing a panel:
docs/agents/adding-a-panel.md.
"""
import sys

from panels.core import registry
from panels.core.context import add_stage_flags, make_ctx
from panels.core.flags import Flags, common_parser
from panels.core.runner import run


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    if argv[0] == 'list':
        for stage in registry.STAGES:
            print(f'{stage}:')
            for n, presets, doc in registry.describe(stage):
                print(f'  {n:24} [{presets}]  {doc}')
        return 0
    if argv[0] == 'check':
        from panels.core.check import check
        problems = check()
        print('\n'.join(problems) or 'all panels ok')
        return 1 if problems else 0
    stage, rest = argv[0], argv[1:]
    if stage == 'finetune':                                         # the stage was renamed (2026-10-27): old command lines keep working
        stage = 'group'
    if stage not in registry.STAGES:
        print(f'unknown stage {stage!r}; use one of {registry.STAGES} or list', file=sys.stderr)
        return 2
    pre = common_parser(stage, add_help=False)
    known, _ = pre.parse_known_args(rest)
    names = registry.select(stage, known.preset, known.panel)
    modules = [registry.load(stage, n) for n in names]

    p = common_parser(stage)
    p.description = __doc__
    add_stage_flags(p, stage)
    flags = Flags(p)
    for m in modules:
        if hasattr(m, 'add_args'):
            m.add_args(flags)
    ctx = make_ctx(stage, p.parse_args(rest), modules, p.error)
    return 1 if run(stage, names, ctx) else 0


if __name__ == '__main__':
    sys.exit(main())
