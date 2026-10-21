"""task_filter_build: fit the task filter of --filter-source (multi, or one dataset) and save it again -> analysis/task_filter/<source>/filter.pt.
Needs the backbone's loso cells on imputed 64-channel gains (docs: model/Qtome/Qtome_filter_tools.py)."""
from tools.analysis.task_filter_eval import filter_context

STAGES = frozenset({'pretrain'})


def run(ctx):
    filter_context(ctx, ctx.args.filter_source, rebuild=True)
