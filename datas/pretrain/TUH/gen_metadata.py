"""
TUH EEG Corpus v2.0.1 (the local subset under raw/tuh_eeg/v2.0.1/edf) -- metadata.json generator.
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), no labels. Clinical routine EEG,
sampling rates 250 / 256 / 400 / 512 Hz (resampled), referential (REF) or linked-ears (LE) montage files.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_INFO = {
    "source_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
    "source_version": "TUH EEG Corpus v2.0.1",
    "file_format": "EDF",
    "description": "Temple University Hospital clinical EEG corpus; routine recordings of mixed patients, no task or labels.",
    "task_type": "Clinical resting EEG",
    "reference": "Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus. Frontiers in "
                 "Neuroscience, 10, 196.",
    "notes": "Subset of 2634 EDF files from 605 patients. Channels = the 22 scalp sites of the 10-20 system found in "
             "every montage (old names T3/T4/T5/T6 resolve to T7/T8/P7/P8 in IO/dataset.py); a file missing one gets a "
             "zero row (padding). Subject ids = 1..N in patient-id order.",
}

# 10-20 sites; coordinates come from MNE's standard_1020 (labels resolve there)
SITES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6",
         "O1", "Oz", "O2", "A1", "A2"]


def main():
    subs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "raw", "tuh_eeg", "v2.0.1", "edf", "*", "*", "*", "*", "*.edf"))):
        subs.setdefault(f.split(os.sep)[-4], []).append(os.path.relpath(f, ROOT))
    meta = {
        "data_metadata": {
            "dataset_name": "TUH", "dataset_info": DATASET_INFO,
            "bids": {"task": None},
            "acquisition": {"sample_frequency": 250, "window_size_seconds": None, "num_subjects": len(subs)},
            "targets": {"count": 1, "type": "pretrain_dummy", "0": {"label": "dummy (continuous window, no task label)"}},
            "channels": {"count": len(SITES), "system": "10-20 International System",
                         **{str(i + 1): {"label": c, "original_label": c.upper()} for i, c in enumerate(SITES)}},
            "cohort": "TUH",
        },
        "data_structure": {str(i + 1): {"patient": p, "runs": [{"file": f} for f in fs]}
                           for i, (p, fs) in enumerate(sorted(subs.items()))},
    }
    with open(os.path.join(ROOT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote TUH/metadata.json: {len(subs)} subjects, {sum(len(v) for v in subs.values())} recordings, {len(SITES)} channels")


if __name__ == "__main__":
    main()
