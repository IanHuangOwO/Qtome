import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader
from IO.tuh import read_sites


class Loader(BaseSubjectLoader):
    """TUAR: every recording of the subject, read at the metadata rate and filtered whole (continuous_transform), cut into
    non-overlapping standard_window (5 s) windows; a window is 1 (artifact) when annotated artifact events (any channel, any type
    but 'null') cover at least half of it, else 0. SCAFFOLDING: match LUNA's TUAR protocol before reporting numbers."""
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(r['file']), self._resolve(r['ann'])) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.sites = [ch[str(i + 1)]['label'] for i in desired_channel_indices]

    @staticmethod
    def _intervals(path):
        out = []
        with open(path) as f:
            for line in f:
                p = line.strip().split(',')
                if len(p) >= 4 and p[1][:1].isdigit() and p[3].lower() != 'null':
                    out.append((float(p[1]), float(p[2])))
        return out

    def _load_data(self):
        present = set(self._existing([p for p, _ in self.runs]))
        data, labels = [], []
        for path, ann in self.runs:
            if path not in present or not os.path.exists(ann):
                continue
            x = read_sites(path, self.sites, self.sample_freq)
            if x is None:
                continue
            sig, sf = self._filter_run(x)
            w = int(round(self.standard_window * sf))
            n = sig.shape[1] // w
            covered = np.zeros(n * w, dtype=bool)               # samples under any artifact event
            for a, b in self._intervals(ann):
                covered[max(0, int(a * sf)):min(n * w, int(b * sf))] = True
            for i in range(n):
                data.append(sig[:, i * w:(i + 1) * w])
                labels.append(int(covered[i * w:(i + 1) * w].mean() >= 0.5))
        if not data:
            return None, None
        return np.stack(data), np.array(labels, dtype=np.int64)
