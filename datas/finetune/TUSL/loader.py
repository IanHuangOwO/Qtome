import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader
from IO.loader import read_sites

LABELS = {'seiz': 0, 'slow': 1, 'bckg': 2, 'cplx': 2}       # complex background counts as background (EEG-FM-Bench: 3 classes)


class Loader(BaseSubjectLoader):
    """TUSL: every recording of the subject, read at the metadata rate and filtered whole (continuous_transform); every
    10 s annotation of its .tse_agg file (start, stop, label, probability) is one window, labelled by LABELS (an unknown label
    is reported and skipped). SCAFFOLDING: check the label strings against the real files."""
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(r['file']), self._resolve(r['ann'])) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.sites = [ch[str(i + 1)]['label'] for i in desired_channel_indices]

    @staticmethod
    def _annotations(path):
        out = []
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and p[0][0].isdigit():
                    out.append((float(p[0]), float(p[1]), p[2].lower()))
        return out

    def _load_data(self):
        present = set(self._existing([p for p, _ in self.runs]))
        data, labels, unknown = [], [], set()
        for path, ann in self.runs:
            if path not in present or not os.path.exists(ann):
                continue
            x = read_sites(path, self.sites, self.sample_freq)
            if x is None:
                continue
            sig, sf = self._filter_run(x)
            w = int(round(self.standard_window * sf))
            for start, stop, name in self._annotations(ann):
                if name not in LABELS:
                    unknown.add(name)
                    continue
                a = int(round(start * sf))
                if a + w <= sig.shape[1]:
                    data.append(sig[:, a:a + w])
                    labels.append(LABELS[name])
        if unknown:
            print(f"  [Warning] TUSL subject {self.subject_id}: unknown labels {sorted(unknown)} skipped")
        if not data:
            return None, None
        return np.stack(data), np.array(labels, dtype=np.int64)
