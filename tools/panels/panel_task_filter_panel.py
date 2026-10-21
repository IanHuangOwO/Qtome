"""task_filter_panel: the filter (--filter-source) applied to each --target dataset it never saw, before and after, at --share
-> analysis/pretrain/task_filter/<source>/panel_<target>_<share>.png. Builds the filter first if it is not saved."""
import os

from tools.analysis.task_filter_eval import filter_context

STAGES = frozenset({'pretrain'})


def run(ctx):
    from tools.viz.task_filter_panel import draw
    if not ctx.args.target:
        raise ValueError('task_filter_panel needs --target <dataset> [...]')
    _, tf = filter_context(ctx, ctx.args.filter_source)
    out = f'{ctx.fig_dir}/task_filter/{ctx.args.filter_source}'; os.makedirs(out, exist_ok=True)
    for dst in ctx.args.target:
        draw(tf, dst, ctx.args.share, f'{out}/panel_{dst}_{ctx.args.share:g}.png')
