"""Helpers shared by the loaders of the TUH EEG corpus and its annotated subsets (datas/pretrain/TUEG, datas/finetune/TUAB, TUEV,
TUSL, TUAR). The recordings are clinical EDF files whose channel names carry a prefix and a reference suffix that vary by
montage file ('EEG FP1-REF', 'EEG FP1-LE'), at 250 / 256 / 400 / 512 Hz."""
import re
import warnings

import numpy as np

NAME = re.compile(r'^EEG\s+([A-Z0-9]+)-(REF|LE|AR\w*|AVG\w*)$', re.I)
# the 21 scalp sites of the 10-20 system every TUH montage has (TUAB's 21 channels; old names T3/T4/T5/T6 resolve to
# T7/T8/P7/P8 in IO/dataset.py)
SITES21 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
           'A1', 'A2']


def rename_channels(raw):
    """'EEG FP1-REF' -> 'FP1' (first occurrence of a site wins); everything else keeps its name."""
    new = {}
    for c in raw.ch_names:
        m = NAME.match(c)
        if m and m.group(1).upper() not in {v.upper() for v in new.values()}:
            new[c] = m.group(1)
    raw.rename_channels(new)
    return raw


def read_sites(path, sites, sample_freq, tmax=None):
    """One EDF recording -> float32 [len(sites), T] in volts at sample_freq (rows of sites the file lacks are zero), None when
    the file is unreadable. tmax: seconds kept from the start."""
    import mne
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            if tmax is not None:
                raw.crop(0, min(tmax, raw.times[-1]), include_tmax=False)
            raw.load_data(verbose=False)
    except Exception as e:
        print(f"  [Warning] {path}: {e}")
        return None
    rename_channels(raw)
    if raw.info['sfreq'] != sample_freq:
        raw.resample(sample_freq, verbose=False)
    names = {c.upper(): c for c in raw.ch_names}
    rows = [i for i, s in enumerate(sites) if s.upper() in names]
    out = np.zeros((len(sites), raw.n_times), dtype=np.float32)
    if rows:
        out[rows] = np.nan_to_num(raw.get_data(picks=[names[sites[i].upper()] for i in rows]))
    return out


def patient_of(path):
    """TUH file name 'aaaaamye_s001_t000.edf' -> the patient id 'aaaaamye'."""
    import os
    return os.path.basename(path).split('_')[0]
