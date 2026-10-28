"""training_dashboard: the finetune training-curve dashboard, drawn from the stored history.json: every fold of every cell of the groups'
backbones under --head (<cell>/finetune/run_<fold>/history.json) -> docs/figures/runs/<backbone>/finetune/<head>/<cell>/<fold>_training_dashboard.png."""
import glob
import os

from IO.config import QLET_OUTPUT
from model.history import History
from panels.core.context import FIGURES
from panels.shared.dashboard import QletDashboard

PRESETS = set()


def draw(ctx):
    for bb in ctx.groups.values():
        for path in sorted(glob.glob(f'{QLET_OUTPUT}/{bb}/finetune/{ctx.head}/*/finetune/run_*/history.json')):
            parts = path.split(os.sep); cell, fold = parts[-4], parts[-2]
            out = f'{FIGURES}/runs/{bb}/finetune/{ctx.head}/{cell}'
            QletDashboard(output_dir=out, history=History.load(path)).plot_finetune(filename=f'{fold}_training_dashboard.png')
