"""task_filter_compare: the multi-task filter (and, with --single <dataset>, that dataset's single-task filter) on each --target with injected artifacts, against
ASR and ICLabel -> analysis/task_filter/compare/compare_<target>.json and .log, analysis/pretrain/task_filter/compare/compare_<target>.png. Builds either filter first if it is not saved."""
import os

from tools.analysis.task_filter_eval import compare, filter_context, write_table

STAGES = frozenset({'pretrain'})


def run(ctx):
    from tools.viz.task_filter_compare import draw
    if not ctx.args.target:
        raise ValueError('task_filter_compare needs --target <dataset> [...]')
    ctxs = {'single-task': filter_context(ctx, ctx.args.single)[1]} if ctx.args.single else {}   # the single-task filter is optional: it lost to the multi-task one
    ctxs['multi-task'] = filter_context(ctx, 'multi')[1]
    sources = ctx.args.filter_sources or list(ctxs['multi-task'].filt['sources'])
    out = f'{ctx.out_dir}/task_filter/compare'; fig = f'{ctx.fig_dir}/task_filter/compare'
    os.makedirs(out, exist_ok=True); os.makedirs(fig, exist_ok=True)
    for dst in ctx.args.target:
        assert dst != ctx.args.single, f'{dst} built the single-task filter: not an unseen target'
        seen = dst in sources                                             # a source of the multi-task filter: in-sample for it, and the figure says so
        lines = []
        log = lambda s: (print(s, flush=True), lines.append(s))
        rows = compare(ctxs, dst, log=log); stem = f'{out}/compare_{dst}'
        write_table(stem, rows, lines)
        draw(rows, dst, {**({'single-task': ctx.args.single} if ctx.args.single else {}), 'multi-task': ' + '.join(sources)}, f'{fig}/compare_{dst}.png', seen=seen)
