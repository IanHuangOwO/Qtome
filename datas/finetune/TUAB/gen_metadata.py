"""
TUAB (TUH Abnormal EEG Corpus v3.0.1) -- metadata.json generator: scans raw/edf/<train|eval>/<normal|abnormal>/<montage>/*.edf.
Normal / abnormal EEG, a label per recording (from its folder). Compass protocol: the first 3 minutes of every recording,
10 s windows, a 10-fold subject split over all patients (no few-shot run). The official train / eval split is kept in each
run's 'split' field. SCAFFOLDING written from the corpus documentation (docs: AAREADME.txt of tuh_eeg_abnormal v3.0.1) before
the data was copied here: run it on the real tree and check the counts against Compass (2,383 subjects, 53,604 windows).
Raw: rsync nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf -> raw/edf (or copy from the 5090's
F:\\tuh_data\\tuh_eeg_abnormal\\v3.0.1\\edf).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import SITES21, patient_of
from IO.metadata import labelled, write_files_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = {'normal': 0, 'abnormal': 1}
CROP_S = 180

DATASET_INFO = {
    "source_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
    "source_version": "tuh_eeg_abnormal v3.0.1",
    "file_format": "EDF",
    "description": "Temple University Hospital Abnormal EEG Corpus: clinical EEG recordings labelled normal or abnormal.",
    "task_type": "Abnormal EEG detection (clinical)",
    "reference": "Lopez, S. (2017). Automated Identification of Abnormal EEGs. Temple University.",
    "notes": "Compass: 2,383 subjects, 21 channels, 250 Hz, 10 s windows, 53,604 windows; first 3 min of each recording; "
             "10-fold subject split. A recording is a pruned segment (t000, t001, ...) of the patient's EEG.",
}


def main(raw_dir=None, root=None):
    raw_dir = raw_dir or os.path.join(ROOT, "raw")
    subs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "edf", "*", "*", "*", "*.edf"))):
        rel = os.path.relpath(f, ROOT) if raw_dir.startswith(ROOT) else f
        split, label = f.split(os.sep)[-4], f.split(os.sep)[-3]
        subs.setdefault(patient_of(f), []).append({"file": rel, "label": LABELS[label], "split": split})
    write_files_metadata(
        root or ROOT, "TUAB", DATASET_INFO,
        acquisition={"sample_frequency": 250, "window_size_seconds": 10.0, "num_subjects": None},
        targets={"count": 2, "type": "abnormal_eeg", "0": {"label": "normal"}, "1": {"label": "abnormal"}},
        channels=labelled(SITES21, str.upper), channel_header={"system": "10-20 International System"},
        structure={str(i + 1): {"patient": p, "runs": runs} for i, (p, runs) in enumerate(sorted(subs.items()))},
        after_info={"crop_seconds": CROP_S}, after_channels={"cohort": "TUEG"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
