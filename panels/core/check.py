"""`python analyze.py check`: every panel imports, has compute and/or draw, a one-line description, valid PRESETS, and its flags can all be
declared together (two panels may share a flag only with the same default). No data, no model: run it after writing or moving a panel."""
import argparse

from panels.core import registry
from panels.core.flags import Flags

KNOWN_PRESETS = {'standard', 'quick'}


def check():
    """Returns a list of problems (empty = fine)."""
    problems = []
    for stage in registry.STAGES:
        flags = Flags(argparse.ArgumentParser(add_help=False))
        presets = set()
        for name in registry.names(stage):
            where = f'{stage}/{name}'
            try:
                m = registry.load(stage, name)
            except Exception as e:
                problems.append(f'{where}: import failed: {e!r}')
                continue
            if not (callable(getattr(m, 'compute', None)) or callable(getattr(m, 'draw', None))):
                problems.append(f'{where}: neither compute nor draw')
            if not (m.__doc__ or '').strip():
                problems.append(f'{where}: no docstring (its first line is the description in `analyze.py list`)')
            bad = set(getattr(m, 'PRESETS', ())) - KNOWN_PRESETS
            if bad:
                problems.append(f'{where}: unknown presets {sorted(bad)}')
            presets |= set(getattr(m, 'PRESETS', ()))
            try:
                if hasattr(m, 'add_args'):
                    m.add_args(flags)
            except Exception as e:
                problems.append(f'{where}: add_args: {e}')
        for p in ('standard',):
            if p not in presets:
                problems.append(f'{stage}: no panel carries the {p!r} preset')
    return problems
