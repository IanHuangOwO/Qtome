"""
TUEV (TUH EEG Events Corpus v2.0.1) -- metadata.json generator: scans raw/edf/train/<patient>/<patient>_<n>.edf and
raw/edf/eval/<idx>/<label>_<idx>_<x>_.edf, each with a sibling .rec annotation file (channel, start s, stop s, code 1-6).
Six event classes (spsw, gped, pled, eyem, artf, bckg). EEG-FM-Bench: 21 channels, 5 s windows, a stratified 0.8 / 0.1 / 0.1
split over all windows (not subject-disjoint). SCAFFOLDING written from the corpus documentation before the data was copied
here: check the window cut against the EEG-FM-Bench / BIOT code before reporting numbers (87,834 / 12,473 / 13,046 windows).
Raw: rsync nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_events/v2.0.1/edf -> raw/edf (or copy from the 5090's
F:\\tuh_data\\tuh_eeg_events\\v2.0.1\\edf).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.tuh import SITES21

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']

DATASET_INFO = {
    "source_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
    "source_version": "tuh_eeg_events v2.0.1",
    "file_format": "EDF + .rec annotations",
    "description": "Temple University Hospital EEG Events Corpus: clinical EEG with six annotated event types.",
    "task_type": "EEG event classification (clinical)",
    "reference": "Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., & Picone, J. (2015). Improved EEG event classification "
                 "using differential energy. IEEE SPMB.",
    "notes": "Train: patient folders; eval: session folders (a session index, not a patient: its subject id is 'eval<idx>'). "
             "EEG-FM-Bench splits all windows at random (0.8 / 0.1 / 0.1), so subject ids only group a patient's files.",
}


def main(raw_dir=None):
    raw_dir = raw_dir or os.path.join(ROOT, "raw")
    subs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "edf", "*", "*", "*.edf"))):
        split, group = f.split(os.sep)[-3], f.split(os.sep)[-2]
        pid = group if split == "train" else f"eval{group}"
        rel = os.path.relpath(f, ROOT) if raw_dir.startswith(ROOT) else f
        subs.setdefault(pid, []).append({"file": rel, "rec": rel[:-4] + ".rec", "split": split})
    meta = {
        "data_metadata": {
            "dataset_name": "TUEV", "dataset_info": DATASET_INFO,
            "acquisition": {"sample_frequency": 250, "window_size_seconds": 5.0, "num_subjects": len(subs)},
            "targets": {"count": 6, "type": "eeg_events", **{str(i): {"label": c} for i, c in enumerate(CLASSES)}},
            "channels": {"count": len(SITES21), "system": "10-20 International System",
                         **{str(i + 1): {"label": c, "original_label": c.upper()} for i, c in enumerate(SITES21)}},
            "cohort": "TUEG",
        },
        "data_structure": {str(i + 1): {"patient": p, "runs": runs} for i, (p, runs) in enumerate(sorted(subs.items()))},
    }
    with open(os.path.join(ROOT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote TUEV/metadata.json: {len(subs)} subjects, {sum(len(v) for v in subs.values())} recordings")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
