"""task_filter_panel: the filter (--filter-source) applied to each --target dataset it never saw, before and after, at --share.
Numbers -> analysis/task_filter/<source>/filter_<target>_<share>.npz, figure -> analysis/pretrain/task_filter/<source>/filter_<target>_<share>.png
(drawn from the saved numbers). Builds the filter first if it is not saved."""
import os

from tools.analysis.task_filter_eval import filter_context

STAGES = frozenset({'pretrain'})


def run(ctx):
    from tools.viz.task_filter_panel import draw, panel_numbers
    if not ctx.args.target:
        raise ValueError('task_filter_panel needs --target <dataset> [...]')
    _, tf = filter_context(ctx, ctx.args.filter_source)
    src, share = ctx.args.filter_source, ctx.args.share
    nd, fd = f'{ctx.out_dir}/task_filter/{src}', f'{ctx.fig_dir}/task_filter/{src}'
    os.makedirs(nd, exist_ok=True); os.makedirs(fd, exist_ok=True)
    for dst in ctx.args.target:
        panel_numbers(tf, dst, share, f'{nd}/filter_{dst}_{share:g}.npz')
        draw(f'{nd}/filter_{dst}_{share:g}.npz', f'{fd}/filter_{dst}_{share:g}.png')
