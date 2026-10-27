"""training_dashboard: the training-curve dashboard, drawn from the stored history.json (the trainers write the numbers, not the plot):
<backbone>/pretrain/artifacts/history.json -> analysis/pretrain/training_dashboard.png."""
import os

from model.history import History
from panels.shared.dashboard import QletDashboard

PRESETS = {'standard'}


def draw(ctx):
    path = os.path.join(os.path.dirname(os.path.dirname(ctx.checkpoint)), 'artifacts', 'history.json')
    if not os.path.exists(path):
        print(f'  skipped: {path} missing (the run predates history.json; its old dashboard, if kept, is in analysis/)')
        return
    QletDashboard(output_dir=ctx.fig_dir, history=History.load(path)).plot_pretrain()
