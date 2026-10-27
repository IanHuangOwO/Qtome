"""training_dashboard: the finetune training-curve dashboard, drawn from the stored history.json: every fold of every cell of the groups'
backbones under --head (<cell>/finetune/run_<fold>/history.json) -> <backbone>/analysis/finetune/<head>/<cell>/<fold>_training_dashboard.png."""
import glob
import os

from IO.config import QTOME_OUTPUT
from model.history import History
from panels.shared.dashboard import QtomeDashboard

PRESETS = set()


def draw(ctx):
    for bb in ctx.groups.values():
        for path in sorted(glob.glob(f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/*/finetune/run_*/history.json')):
            parts = path.split(os.sep); cell, fold = parts[-4], parts[-2]
            out = f'{QTOME_OUTPUT}/{bb}/analysis/finetune/{ctx.head}/{cell}'
            QtomeDashboard(output_dir=out, history=History.load(path)).plot_finetune(filename=f'{fold}_training_dashboard.png')
