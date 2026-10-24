"""task_filter_filter: the filter of --filter-source alone, applied to nothing, at --share: power kept of white input, effective rank and the
importance of each channel for each Q-atom. Numbers -> analysis/task_filter/<source>/filter_analysis_<share>.npz, figure ->
analysis/pretrain/task_filter/<source>/filter_analysis_<share>.png (drawn from the saved numbers). Builds the filter first if it is not saved."""
import os

from tools.analysis.task_filter_eval import filter_context

STAGES = frozenset({'pretrain'})


def run(ctx):
    from tools.viz.task_filter_panel import draw_filter, filter_numbers
    _, tf = filter_context(ctx, ctx.args.filter_source)
    src, share = ctx.args.filter_source, ctx.args.share
    nd, fd = f'{ctx.out_dir}/task_filter/{src}', f'{ctx.fig_dir}/task_filter/{src}'
    os.makedirs(nd, exist_ok=True); os.makedirs(fd, exist_ok=True)
    filter_numbers(tf, share, f'{nd}/filter_analysis_{share:g}.npz')
    draw_filter(f'{nd}/filter_analysis_{share:g}.npz', f'{fd}/filter_analysis_{share:g}.png')
