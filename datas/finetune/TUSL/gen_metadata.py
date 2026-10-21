"""
TUSL (TUH EEG Slowing Corpus v2.0.1) -- metadata.json generator: scans raw/edf/**/<stem>.edf with a sibling <stem>.tse_agg
(term-based annotations: start s, stop s, label, probability; every annotation 10 s). Three classes seizure / slowing /
background (complex background counted as background). EEG-FM-Bench: 21-22 channels, 10 s windows, 210 / 43 / 37, stratified
over all windows. SCAFFOLDING written from the corpus documentation before the data was copied here: check the label strings
of the .tse_agg files and the class counts against EEG-FM-Bench before reporting numbers.
Raw: rsync nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_slowing/v2.0.1/edf -> raw/edf (or copy from the 5090's
F:\\tuh_data\\tuh_eeg_slowing\\v2.0.1\\edf).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.tuh import SITES21, patient_of

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['seizure', 'slowing', 'background']

DATASET_INFO = {
    "source_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
    "source_version": "tuh_eeg_slowing v2.0.1",
    "file_format": "EDF + .tse_agg annotations",
    "description": "Temple University Hospital EEG Slowing Corpus: 10 s samples of seizure, slowing and (complex) background.",
    "task_type": "Slowing / seizure / background classification (clinical)",
    "reference": "von Weltin, E., Ahsan, T., Shah, V., Jamshed, D., Golmohammadi, M., Obeid, I., & Picone, J. (2017). Electroencephalographic "
                 "slowing: a primary source of error in automatic seizure detection. IEEE SPMB.",
    "notes": "Seizure samples come from the TUH EEG Seizure Corpus, slowing and complex background were collected manually.",
}


def main(raw_dir=None):
    raw_dir = raw_dir or os.path.join(ROOT, "raw")
    subs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "edf", "**", "*.edf"), recursive=True)):
        ann = f[:-4] + ".tse_agg"
        if not os.path.exists(ann):
            continue
        rel = os.path.relpath(f, ROOT) if raw_dir.startswith(ROOT) else f
        subs.setdefault(patient_of(f), []).append({"file": rel, "ann": rel[:-4] + ".tse_agg"})
    meta = {
        "data_metadata": {
            "dataset_name": "TUSL", "dataset_info": DATASET_INFO,
            "acquisition": {"sample_frequency": 250, "window_size_seconds": 10.0, "num_subjects": len(subs)},
            "targets": {"count": 3, "type": "slowing", **{str(i): {"label": c} for i, c in enumerate(CLASSES)}},
            "channels": {"count": len(SITES21), "system": "10-20 International System",
                         **{str(i + 1): {"label": c, "original_label": c.upper()} for i, c in enumerate(SITES21)}},
            "cohort": "TUEG",
        },
        "data_structure": {str(i + 1): {"patient": p, "runs": runs} for i, (p, runs) in enumerate(sorted(subs.items()))},
    }
    with open(os.path.join(ROOT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote TUSL/metadata.json: {len(subs)} patients, {sum(len(v) for v in subs.values())} recordings")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
