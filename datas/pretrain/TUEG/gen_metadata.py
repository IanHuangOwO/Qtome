"""
TUH EEG Corpus (TUEG) v2.0.2 (raw/tuh_eeg/v2.0.2 -> the Expansion drive; a seeded subset of it is compiled) -- metadata.json generator.
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), no labels. Clinical routine EEG,
sampling rates 250 / 256 / 400 / 512 Hz (resampled), referential (REF) or linked-ears (LE) montage files.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
N_PATIENTS, MIN_SECONDS, SEED = 1000, 300, 0     # the corpus rule: a dataset over 1,000 subjects keeps a seeded pick of 1,000

DATASET_INFO = {
    "source_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
    "source_version": "TUH EEG Corpus v2.0.2",
    "file_format": "EDF",
    "description": "Temple University Hospital clinical EEG corpus; routine recordings of mixed patients, no task or labels.",
    "task_type": "Clinical resting EEG",
    "reference": "Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus. Frontiers in "
                 "Neuroscience, 10, 196.",
    "notes": "Seeded subset: 1000 patients, one recording each (>= 300 s long). Channels = the 22 scalp sites of the 10-20 system found in "
             "every montage (old names T3/T4/T5/T6 resolve to T7/T8/P7/P8 in IO/dataset.py); a file missing one gets a "
             "zero row (padding). Subject ids = 1..N in patient-id order.",
}

# 10-20 sites; coordinates come from MNE's standard_1020 (labels resolve there)
SITES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6",
         "O1", "Oz", "O2", "A1", "A2"]


def duration(path):                                   # seconds, from the EDF header (records x record duration)
    with open(path, "rb") as f:
        h = f.read(256)
    return int(h[236:244]) * float(h[244:252])


def main():
    by_patient = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "raw", "tuh_eeg", "v2.0.2", "edf", "*", "*", "*", "*", "*.edf"))):
        by_patient.setdefault(f.split(os.sep)[-4], []).append(f)
    rng = np.random.default_rng(SEED)
    subs = {}
    for p in rng.permutation(sorted(by_patient)):
        for f in rng.permutation(by_patient[p]):
            if duration(f) >= MIN_SECONDS:
                subs[str(p)] = [os.path.relpath(f, ROOT)]; break
        if len(subs) == N_PATIENTS:
            break
    meta = {
        "data_metadata": {
            "dataset_name": "TUEG", "dataset_info": DATASET_INFO,
            "bids": {"task": None},
            "acquisition": {"sample_frequency": 250, "window_size_seconds": None, "num_subjects": len(subs)},
            "targets": {"count": 1, "type": "pretrain_dummy", "0": {"label": "dummy (continuous window, no task label)"}},
            "channels": {"count": len(SITES), "system": "10-20 International System",
                         **{str(i + 1): {"label": c, "original_label": c.upper()} for i, c in enumerate(SITES)}},
            "cohort": "TUEG",
        },
        "data_structure": {str(i + 1): {"patient": p, "runs": [{"file": f} for f in fs]}
                           for i, (p, fs) in enumerate(sorted(subs.items()))},
    }
    with open(os.path.join(ROOT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote TUEG/metadata.json: {len(subs)} subjects, {sum(len(v) for v in subs.values())} recordings, {len(SITES)} channels")


if __name__ == "__main__":
    main()
