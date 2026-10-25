"""task_filter_eval: tests of the filter (--filter-source) on unseen datasets, --filter-test mi / p300 / noise (default all three) on --target
(default per test) -> pretrain/analysis/filter/<source>/eval_<test>_<target>.json and .log. Builds the filter first if it is not saved."""
from panels.shared.task_filter.context import filter_context, write_table
from panels.shared.task_filter.tests import DEFAULT_TARGETS, TESTS

PRESETS = set()


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: task_filter/fit.py SOURCES)')
    flags.add('--filter-test', nargs='+', choices=['mi', 'p300', 'noise'], help='tests (default all)')
    flags.add('--target', nargs='+', help='unseen datasets (default per test)')


def compute(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    for test in ctx.args.filter_test or list(TESTS):
        for dst in ctx.args.target or DEFAULT_TARGETS[test]:
            lines = []
            log = lambda s: (print(s, flush=True), lines.append(s))
            write_table(f'{out}/eval_{test}_{dst}', TESTS[test](tf, dst, log=log), lines)
