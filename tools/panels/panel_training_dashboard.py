"""training_dashboard: the training-curve dashboard, drawn from the stored history.json (the trainers write the numbers, not the plot).
pretrain: <backbone>/pretrain/artifacts/history.json -> analysis/pretrain/training_dashboard.png.
finetune: every fold of every cell of the groups' backbones under --head (<cell>/finetune/run_<fold>/history.json)
-> <backbone>/analysis/finetune/<head>/<cell>/<fold>_training_dashboard.png."""
import glob
import os

from model.history import History
from model.factory import MODEL_REGISTRY
from tools.analysis import QTOME_OUTPUT

STAGES = frozenset({'pretrain', 'finetune'})


def run(ctx):
    plotter_cls = MODEL_REGISTRY['Qtome'].plotter_cls
    if ctx.stage == 'pretrain':
        path = os.path.join(os.path.dirname(os.path.dirname(ctx.checkpoint)), 'artifacts', 'history.json')
        if not os.path.exists(path):
            print(f'  skipped: {path} missing (the run predates history.json; its old dashboard, if kept, is in analysis/)')
            return
        plotter_cls(output_dir=ctx.fig_dir, history=History.load(path)).plot_pretrain()
        return
    for bb in ctx.groups.values():
        for path in sorted(glob.glob(f'{QTOME_OUTPUT}/{bb}/finetune/{ctx.head}/*/finetune/run_*/history.json')):
            parts = path.split(os.sep); cell, fold = parts[-4], parts[-2]
            out = f'{QTOME_OUTPUT}/{bb}/analysis/finetune/{ctx.head}/{cell}'
            plotter_cls(output_dir=out, history=History.load(path)).plot_finetune(
                filename=f'{fold}_training_dashboard.png')
