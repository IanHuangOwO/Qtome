"""
Helsinki neonatal EEG with seizure annotations (Stevenson et al. 2019), Zenodo record
4940267 -- metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.
Fetch with ./fetch.sh (resumable).

Source verified (2026-09-24) against the Zenodo record and the EDF headers: 79 term
neonates from the Helsinki University Hospital NICU, one EDF per neonate (eegN.edf),
256 Hz, median 74 min. Channels are 'EEG <site>-Ref' for 19 10-20 electrodes plus
'ECG EKG' and 'Resp Effort', which are left out. T3/T4/T5/T6 are relabelled
T7/T8/P7/P8.

Used for SELF-SUPERVISED PRETRAINING ONLY: each recording is cut into non-overlapping
5 s windows with dummy label 0. The three experts' seizure annotations
(annotations_2017_{A,B,C}.csv, one row per second) are not read. Neonatal EEG differs
a lot from adult EEG (slower rhythms, discontinuous background), so check this
dataset's effect on the adult benchmarks before keeping it in the pretrain mix.
"""
import glob
import os
import re
from collections import OrderedDict
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.metadata import write_files_metadata

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://zenodo.org/records/4940267",
    "file_format": "EDF",
    "description": (
        "79 term neonates in the NICU, 19-channel 10-20 EEG at 256 Hz, median 74 min "
        "per recording; seizures annotated by three experts (not used here)."
    ),
    "task_type": "pretrain_dummy",
    "reference": "Stevenson NJ, Tapani K, Lauronen L, Vanhatalo S (2019). A dataset of "
                 "neonatal EEG recordings with seizure annotations. Sci Data 6:190039. "
                 "doi:10.1038/sdata.2019.39. Zenodo doi:10.5281/zenodo.4940267",
    "notes": "Pretraining only: 5 s non-overlapping windows, dummy label 0; seizure annotations ignored.",
}

_CH = [("Fp1", "Fp1"), ("Fp2", "Fp2"), ("F3", "F3"), ("F4", "F4"), ("F7", "F7"),
       ("F8", "F8"), ("Fz", "Fz"), ("C3", "C3"), ("C4", "C4"), ("Cz", "Cz"),
       ("T7", "T3"), ("P7", "T5"), ("T8", "T4"), ("P8", "T6"), ("P3", "P3"),
       ("P4", "P4"), ("Pz", "Pz"), ("O1", "O1"), ("O2", "O2")]
CHANNELS = OrderedDict((str(i + 1), {"label": lab, "original_label": f"EEG {raw}-Ref"})
                       for i, (lab, raw) in enumerate(_CH))

TARGETS = {
    "count": 1,
    "type": "pretrain_dummy",
    "0": {"label": "dummy (continuous EEG window, no task label)"},
}


def main():
    files = glob.glob(os.path.join(ROOT, "raw", "eeg*.edf"))
    num = lambda p: int(re.match(r"eeg(\d+)\.edf$", os.path.basename(p)).group(1))
    structure = OrderedDict((str(num(p)), {"file": "raw/" + os.path.basename(p)})
                            for p in sorted(files, key=num))
    write_files_metadata(
        ROOT, "Neonatal_Helsinki", DATASET_INFO,
        acquisition={'sample_frequency': 256, 'window_size_seconds': 5.0, 'num_subjects': None},
        targets=TARGETS,
        channels=CHANNELS,
        structure=structure,
        channel_header={'system': "10-20"},
        newline=False)


if __name__ == "__main__":
    main()
