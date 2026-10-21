import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader
from IO.tuh import read_sites


class Loader(BaseSubjectLoader):
    """TUEV: every recording of the subject, whole recording read at the metadata rate and filtered (continuous_transform); one
    standard_window (5 s) window centred on each distinct annotated event of its .rec file (channel,start,stop,code; the
    channel is ignored, an event is distinct by (start, stop, code)), labelled code - 1. Events whose window leaves the
    recording are dropped. SCAFFOLDING: EEG-FM-Bench / BIOT may cut the windows differently."""
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(r['file']), self._resolve(r['rec'])) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.sites = [ch[str(i + 1)]['label'] for i in desired_channel_indices]

    @staticmethod
    def _events(rec_path):
        out = set()
        with open(rec_path) as f:
            for line in f:
                p = line.strip().split(',')
                if len(p) == 4:
                    out.add((float(p[1]), float(p[2]), int(p[3]) - 1))
        return sorted(out)

    def _load_data(self):
        present = set(self._existing([p for p, _ in self.runs]))
        data, labels = [], []
        for path, rec in self.runs:
            if path not in present or not os.path.exists(rec):
                continue
            x = read_sites(path, self.sites, self.sample_freq)
            if x is None:
                continue
            sig, sf = self._filter_run(x)
            w = int(round(self.standard_window * sf))
            for start, stop, label in self._events(rec):
                a = int(round((start + stop) / 2 * sf)) - w // 2
                if 0 <= a and a + w <= sig.shape[1]:
                    data.append(sig[:, a:a + w])
                    labels.append(label)
        if not data:
            return None, None
        return np.stack(data), np.array(labels, dtype=np.int64)
