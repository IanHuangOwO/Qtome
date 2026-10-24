"""Panel flags. A panel declares its own options in add_args(flags); several panels may declare the same option (--max-trials,
--datasets): the first declaration wins, a later one must agree on the default or it is an error."""
import argparse


class Flags:
    def __init__(self, parser):
        self._p, self._seen = parser, {}

    def add(self, *names, **kw):
        key = names[0]
        if key in self._seen:
            if self._seen[key] != kw.get('default'):
                raise ValueError(f"flag {key} declared twice with different defaults: {self._seen[key]!r} vs {kw.get('default')!r}")
            return
        self._seen[key] = kw.get('default')
        self._p.add_argument(*names, **kw)


def common_parser(stage, add_help=True):
    p = argparse.ArgumentParser(prog=f'analyze.py {stage}', add_help=add_help,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default='standard', help='panels run by default: the ones tagged with this preset')
    p.add_argument('--panel', action='append', default=[], help='run these panels instead of the preset (repeatable)')
    p.add_argument('--redraw', action='store_true', help='draw only, from the numbers saved by an earlier run')
    return p
