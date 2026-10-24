"""Panel discovery. A panel is panels/<stage>/<name>.py, or a folder panels/<stage>/<name>/ whose __init__.py is the panel.
Adding a panel is adding a file: nothing central lists it. A panel module defines

    PRESETS = {'standard', 'quick'}     which presets run it (empty or absent: on request only)
    NEEDS_CHECKPOINT = True             (pretrain) False when it runs without a trained backbone, e.g. profile
    add_args(flags)                     optional: this panel's own options (flags.add('--share', ...))
    compute(ctx)                        optional: write numbers under ctx.numbers_dir
    draw(ctx)                           optional: read those numbers, write figures under ctx.fig_dir

and the docstring's first line is its description in `analyze.py list`."""
import importlib
import os

STAGES = ('pretrain', 'finetune')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def names(stage):
    d = os.path.join(_ROOT, stage)
    out = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.startswith('_'):
            continue
        if f.endswith('.py'):
            out.append(f[:-3])
        elif os.path.isdir(p) and os.path.exists(os.path.join(p, '__init__.py')):
            out.append(f)
    return out


def load(stage, name):
    if name not in names(stage):
        raise ValueError(f'no {stage} panel named {name!r}; available: {names(stage)}')
    return importlib.import_module(f'panels.{stage}.{name}')


def select(stage, preset, wanted):
    """--panel names if given, else the panels tagged with the preset."""
    if wanted:
        for n in wanted:
            load(stage, n)
        return list(wanted)
    chosen = [n for n in names(stage) if preset in getattr(load(stage, n), 'PRESETS', ())]
    if not chosen:
        raise ValueError(f'no {stage} panel carries the preset {preset!r}')
    return chosen


def describe(stage):
    rows = []
    for n in names(stage):
        m = load(stage, n)
        doc = (m.__doc__ or '').strip().splitlines()
        rows.append((n, ','.join(sorted(getattr(m, 'PRESETS', ()))) or '-', doc[0] if doc else ''))
    return rows
