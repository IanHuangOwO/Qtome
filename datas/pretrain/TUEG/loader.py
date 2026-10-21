from IO.loader import BidsLoader
from IO.tuh import rename_channels


class Loader(BidsLoader):
    """TUH EEG Corpus v2.0.1: EDF recordings cut into continuous windows (BidsLoader). Channel names carry a prefix and a
    reference suffix that vary by montage file ('EEG FP1-LE', 'EEG FP1-REF'); IO/tuh.py strips them so the metadata labels
    match. Non-scalp channels (EKG, PHOTIC, numbered aux) match nothing and are dropped."""
    def _prepare(self, raw):
        return rename_channels(raw)
