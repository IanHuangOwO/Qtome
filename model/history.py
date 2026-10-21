"""History: the per-epoch numbers of a training run, kept by the trainers and stored as history.json. No plotting here:
training imports this module only; the dashboard is drawn afterwards from the file (model/base_plotter.py, the
training_dashboard panel)."""
import json
import os


class History:
    """Per-epoch training / validation metrics, kept by the trainers and stored as history.json (the numbers; the
    dashboard is drawn from them afterwards by the training_dashboard panel -- a BasePlotter built with history=)."""

    def __init__(self):
        self.history = {'train': {}, 'val': {}}

    def update(self, train_metrics, val_metrics=None, epoch_info=None):
        for k, v in train_metrics.items():
            self.history['train'].setdefault(k, []).append(v)
        if val_metrics:
            for k, v in val_metrics.items():
                self.history['val'].setdefault(k, []).append(v)
        if epoch_info is not None:                           # one dict per epoch: epoch, lr, seconds, phase, curriculum, ...
            self.history.setdefault('epochs', []).append(epoch_info)

    def save(self, path):                                    # write-then-rename: a reader never sees a half file
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.history, f, default=float)
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        with open(path) as f:
            return json.load(f)
