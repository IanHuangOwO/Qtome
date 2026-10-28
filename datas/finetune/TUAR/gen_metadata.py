"""
TUAR (TUH EEG Artifact Corpus v3.0.1) -- metadata.json generator: scans raw/edf/<montage>/<file>.edf with a sibling <file>.csv
(channel, start s, stop s, label, confidence; the label an artifact type: eyem, chew, shiv, musc, elpp, elec, or null).
Artifact detection: a 5 s window is positive when annotated artifacts cover at least half of it. SCAFFOLDING written from the
corpus documentation before the data was copied here: LUNA's TUAR protocol (windowing, label rule, split) must be matched
before reporting numbers (not a Compass or EEG-FM-Bench dataset; also a test bed for the task filter).
Raw: rsync nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_artifact/v3.0.1/edf -> raw/edf (or copy from the 5090's
F:\\tuh_data\\tuh_eeg_artifact\\v3.0.1\\edf).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import SITES21, patient_of

from IO.metadata import write_files_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_INFO = {
    "source_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
    "source_version": "tuh_eeg_artifact v3.0.1",
    "file_format": "EDF + csv annotations",
    "description": "Temple University Hospital EEG Artifact Corpus: clinical EEG with channel-wise artifact annotations.",
    "task_type": "Artifact detection (clinical)",
    "reference": "Buckwalter, G., Chhin, S., Rahman, S., Obeid, I., & Picone, J. (2021). Recent advances in the TUH EEG corpus: "
                 "improving the interrater agreement for artifacts and epileptiform events. IEEE SPMB.",
    "notes": "Annotations use the TCP bipolar montage; a window's label here ignores the channel and artifact type.",
}


def main(raw_dir=None):
    raw_dir = raw_dir or os.path.join(ROOT, "raw")
    subs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "edf", "*", "*.edf"))):
        ann = f[:-4] + ".csv"
        if not os.path.exists(ann):
            continue
        rel = os.path.relpath(f, ROOT) if raw_dir.startswith(ROOT) else f
        subs.setdefault(patient_of(f), []).append({"file": rel, "ann": rel[:-4] + ".csv"})
    write_files_metadata(
        ROOT, "TUAR", DATASET_INFO,
        acquisition={'sample_frequency': 250, 'window_size_seconds': 5.0, 'num_subjects': None},
        targets={"count": 2, "type": "artifact_detection", "0": {"label": "clean"}, "1": {"label": "artifact"}},
        channels={str(i + 1): {"label": c, "original_label": c.upper()} for i, c in enumerate(SITES21)},
        structure={str(i + 1): {"patient": p, "runs": runs} for i, (p, runs) in enumerate(sorted(subs.items()))},
        channel_header={'system': "10-20 International System"},
        after_channels={'cohort': "TUEG"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
