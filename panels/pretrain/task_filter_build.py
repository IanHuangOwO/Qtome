"""task_filter_build: fit the task filter of --filter-source (multi, or one dataset) and save it again -> analysis/task_filter/<source>/filter.pt.
Needs the backbone's loso cells on imputed 64-channel gains (docs: model/Qtome/Qtome_filter_tools.py)."""
from panels.shared.task_filter.context import filter_context

PRESETS = set()


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: Qtome_filter_tools SOURCES)')


def compute(ctx):
    filter_context(ctx, ctx.args.filter_source, rebuild=True)
