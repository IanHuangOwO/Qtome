import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader
from IO.loader import read_sites


class Loader(BaseSubjectLoader):
    """TUAB: every recording of the patient, its first crop_seconds (180) read at the metadata rate, the 21 sites picked by name
    (a missing one zero), the whole crop filtered (continuous_transform) and cut into non-overlapping standard_window (10 s)
    windows, each labelled with its recording's normal / abnormal label."""
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(r['file']), r['label']) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.sites = [ch[str(i + 1)]['label'] for i in desired_channel_indices]
        self.crop = self.data_metadata.get('crop_seconds')

    def _load_data(self):
        present = set(self._existing([p for p, _ in self.runs]))
        data, labels = [], []
        for path, label in self.runs:
            if path not in present:
                continue
            x = read_sites(path, self.sites, self.sample_freq, tmax=self.crop)
            if x is None:
                continue
            sig, sf = self._filter_run(x)
            w = int(round(self.standard_window * sf))
            for i in range(sig.shape[1] // w):
                data.append(sig[:, i * w:(i + 1) * w])
                labels.append(label)
        if not data:
            return None, None
        return np.stack(data), np.array(labels, dtype=np.int64)
