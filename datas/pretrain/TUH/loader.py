import re

from IO.loader import BidsLoader

_NAME = re.compile(r'^EEG\s+([A-Z0-9]+)-(REF|LE|AR\w*|AVG\w*)$', re.I)       # 'EEG FP1-REF' / 'EEG FP1-LE'


class Loader(BidsLoader):
    """TUH EEG Corpus v2.0.1: EDF recordings cut into continuous windows (BidsLoader). Channel names carry a prefix and a
    reference suffix that vary by montage file ('EEG FP1-LE', 'EEG FP1-REF'); the prefix and suffix are stripped so the
    metadata labels match. Non-scalp channels (EKG, PHOTIC, numbered aux) match nothing and are dropped."""
    def _prepare(self, raw):
        new = {}
        for c in raw.ch_names:
            m = _NAME.match(c)
            if m and m.group(1).upper() not in {v.upper() for v in new.values()}:
                new[c] = m.group(1)
        raw.rename_channels(new)
        return raw
