"""task_filter_compare: the single-task filter of --single and the multi-task filter on each --target with injected artifacts, against
ASR and ICLabel -> analysis/task_filter/compare/compare_<target>.json and .log, analysis/pretrain/task_filter/compare/compare_<target>.png. Builds either filter first if it is not saved."""
import os

from tools.analysis.task_filter_eval import compare, filter_context, write_table

STAGES = frozenset({'pretrain'})


def run(ctx):
    from tools.viz.task_filter_compare import draw
    if not (ctx.args.target and ctx.args.single):
        raise ValueError('task_filter_compare needs --target <dataset> [...] and --single <source dataset>')
    ctxs = {'single-task': filter_context(ctx, ctx.args.single)[1], 'multi-task': filter_context(ctx, 'multi')[1]}
    sources = ctx.args.filter_sources or list(ctxs['multi-task'].filt['sources'])
    out = f'{ctx.out_dir}/task_filter/compare'; fig = f'{ctx.fig_dir}/task_filter/compare'
    os.makedirs(out, exist_ok=True); os.makedirs(fig, exist_ok=True)
    for dst in ctx.args.target:
        assert dst != ctx.args.single and dst not in sources, f'{dst} helped build a filter: not an unseen target'
        lines = []
        log = lambda s: (print(s, flush=True), lines.append(s))
        rows = compare(ctxs, dst, log=log); stem = f'{out}/compare_{dst}'
        write_table(stem, rows, lines)
        draw(rows, dst, {'single-task': ctx.args.single, 'multi-task': ' + '.join(sources)}, f'{fig}/compare_{dst}.png')
