"""task_filter_eval: tests of the filter (--filter-source) on unseen datasets, --filter-test mi / p300 / noise (default all three) on --target
(default per test) -> analysis/task_filter/<source>/eval_<test>_<target>.json and .log. Builds the filter first if it is not saved."""
from tools.analysis.task_filter_eval import DEFAULT_TARGETS, TESTS, filter_context, write_table

STAGES = frozenset({'pretrain'})


def run(ctx):
    out, tf = filter_context(ctx, ctx.args.filter_source)
    for test in ctx.args.filter_test or list(TESTS):
        for dst in ctx.args.target or DEFAULT_TARGETS[test]:
            lines = []
            log = lambda s: (print(s, flush=True), lines.append(s))
            write_table(f'{out}/eval_{test}_{dst}', TESTS[test](tf, dst, log=log), lines)
