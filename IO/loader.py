import json
import os
import re
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

from IO.channels import get_standard_coords
from IO.preprocessing import cut_event_window, cut_windows


class BaseSubjectLoader(ABC):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        self.config = config
        self.subject_id = subject_id
        self.channel_indices = desired_channel_indices

        self.data_metadata = config['data_metadata']
        self.dataset_params = config['dataset_params']
        self.data_root = self.dataset_params['dataset_path']

        targets = self.data_metadata['targets']
        self.num_targets = targets.get('count', 2)
        self.target_type = targets.get('type', 'motor_imagery')

        acquisition = self.data_metadata['acquisition']
        self.sample_freq = acquisition['sample_frequency']
        self.standard_window = acquisition.get('window_size_seconds', None)
        self.target_points = int(self.standard_window * self.sample_freq) if self.standard_window else None
        # Global compile-time setting (configs/compile.json's compile_params, passed into
        # every loader's dataset_params by cache_dataset.py -- same mechanism dataset_path
        # already uses). 0.0 default = old behavior (window starts exactly at the event,
        # zero-length post) for every loader that hasn't opted in. Only meaningful for a
        # trigger/annotation-anchored loader that actually reads it (BCICIV1_Train,
        # BNCI2014001, Inria_Train as of this comment)
        # for the per-dataset headroom measured before enabling this. In NATIVE sample-rate
        # samples wherever a loader converts to pts (self.sample_freq, not the compile
        # target) -- cache_dataset.py rescales the resulting valid_ranges to the compiled
        # rate itself, see get_subject_data.
        # cache_dataset.py sets continuous_transform (its bandpass + resample): a loader that can
        # apply it to each whole recording BEFORE cutting epochs does so and sets prefiltered, and
        # the cache then skips filtering the cut epochs (filtering short epochs leaves edge
        # transients: ~27% error on 1 s epochs). valid_ranges are then compiled-rate indices.
        self.continuous_transform = None
        self.prefiltered = False
        self.pre_event_seconds = self.dataset_params.get('pre_event_seconds', 0.0)
        self.post_event_seconds = self.dataset_params.get('post_event_seconds', 0.0)
        # continuous_seconds (compile_params, event-free experiment): an event-cutting loader that supports it cuts
        # every recording into non-overlapping windows of this length instead (MoabbLoader, Inria, BCIC2020-3)
        self.continuous_seconds = self.dataset_params.get('continuous_seconds', 0.0)
        if self.continuous_seconds:   # fixed-window loaders cut continuous_seconds windows (30 s runs), not the metadata's
            self.standard_window = self.continuous_seconds
            self.target_points = int(self.standard_window * self.sample_freq)
        # Set by a loader's _load_data() when it cuts event windows: a list of N (start, end)
        # sample pairs (native rate, or compiled rate once prefiltered), one per trial this subject
        # produced, marking real (non-padded) content -- see IO/preprocessing.py's
        # cut_event_window. None (default, and every loader that never sets this) means
        # "every trial fully real", read by get_subject_data below.
        self._last_valid_ranges = None
        # Optional per-trial session index (0 = the subject's first session, in recording
        # order), set by loaders that know it (MoabbLoader). None = every trial session 0.
        self._last_sessions = None

    def _continuous_windows(self, sig, sf):
        """sig [C, T] at rate sf -> list of non-overlapping continuous_seconds windows [C, win] (the tail is dropped)."""
        win = int(round(self.continuous_seconds * sf))
        n = sig.shape[1] // win
        return list(sig[:, :n * win].reshape(sig.shape[0], n, win).transpose(1, 0, 2)) if n else []

    def _require_subject(self, subject_id) -> Dict:
        """Looks up this subject's data_structure entry, raising a clear error if missing."""
        subject_str = str(subject_id)
        structure = self.config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {subject_id} not found in data structure.")
        return structure[subject_str]

    def _resolve(self, rel_path: str) -> str:
        """Joins a data_structure-relative path onto this dataset's root."""
        return os.path.join(self.data_root, rel_path.lstrip('./'))

    def _resample_if_needed(self, raw) -> None:
        if raw.info['sfreq'] != self.sample_freq:
            raw.resample(self.sample_freq)

    @staticmethod
    def _existing(paths: List[str]) -> List[str]:
        """Filters to paths that exist, printing a warning for each one that doesn't."""
        for p in paths:
            if not os.path.exists(p):
                print(f"  [Warning] Missing file: {p}")
        return [p for p in paths if os.path.exists(p)]

    def _filter_run(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """One continuous recording [C, T] at self.sample_freq -> (recording, its sample rate).
        With cache_dataset.py's continuous_transform set, the WHOLE recording is bandpassed and
        resampled here, before any cutting, and the loader is marked prefiltered (cut windows and
        valid_ranges are then in compiled-rate samples)."""
        tf = self.continuous_transform
        if tf is None:
            return data, self.sample_freq
        self.prefiltered = True
        return tf(data).numpy(), tf.sample_freq

    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_subject_data(self) -> Optional[Dict[str, Any]]:
        self._last_valid_ranges = self._last_sessions = None  # reset; an event-cutting loader sets them
        data, labels = self._load_data()
        if data is None:
            return None
        n = len(data)
        valid_ranges = self._last_valid_ranges if self._last_valid_ranges is not None \
            else [(0, data.shape[-1])] * n
        return {
            'data': data.astype(np.float32),
            'labels': labels.astype(np.int64),
            'subject_id': self.subject_id,
            # NATIVE-rate (self.sample_freq) sample indices -- cache_dataset.py rescales
            # to the compiled rate before saving (see BandpassResample changing sample
            # count). [(0, T)]*n (every trial fully real) for every loader that doesn't
            # set self._last_valid_ranges -- i.e. all of them except the event-anchored
            # ones (see IO/preprocessing.py's cut_event_window).
            'valid_ranges': valid_ranges,
            'prefiltered': self.prefiltered,
            'session': np.asarray(self._last_sessions if self._last_sessions is not None else [0] * n,
                                  dtype=np.int64),
        }


def resolve_dataset_loader(dataset_path: str):
    """Dynamically imports datas/<Name>/loader.py's `Loader` class — used only
    by cache_dataset.py; train-time reads the compiled cache directly in
    IO/dataset.py, no loader class involved. Directory presence of loader.py
    IS the registration; there is no separate registry to keep in sync. A MOABB
    dataset needs no loader.py: its metadata.json has the data_metadata.moabb
    block, which selects MoabbLoader itself."""
    import importlib.util
    mod_path = os.path.join(dataset_path, 'loader.py')
    if not os.path.exists(mod_path):
        meta_path = os.path.join(dataset_path, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                if 'moabb' in json.load(f).get('data_metadata', {}):
                    return MoabbLoader
        raise FileNotFoundError(f"No {mod_path} — see docs/agents/adding-a-dataset.md Step 6.")
    spec = importlib.util.spec_from_file_location(f'dataset_loader_{os.path.basename(dataset_path)}', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Loader


# --- MOABB-backed loading ------------------------------------------------------------
# MOABB-backed dataset loading (compile-time only, like every datas/<Name>/loader.py).
#
# For a dataset MOABB covers there is no datas/<Name>/loader.py (resolve_dataset_loader picks MoabbLoader from the
# metadata's moabb block) and gen_metadata.py calls write_moabb_metadata() (IO/metadata.py). MOABB does the download and the file
# parsing (events, sessions, runs); everything after -- trial cutting with the global
# pre/post_event_seconds, bandpass/resample, channel unification -- stays ours
# (cache_dataset.py). MOABB's paradigm/evaluation layers are deliberately not used.
#
# Raw files live in datas/<Name>/raw/ in MOABB's own layout (MNE-<code>-data/...):
# moabb_dataset() redirects MOABB's storage-path lookup there.

_RAW_DIR = None


def _get_dataset_path(sign, path=None):
    return path if path is not None else _RAW_DIR


def moabb_dataset(class_name: str, dataset_path: str, raw_root: str = None, **kwargs):
    """MOABB dataset whose downloads land in <dataset_path>/raw/. raw_root (metadata moabb.raw_root, relative to
    dataset_path): read <raw_root>/raw/ instead -- several dataset folders sharing one download, e.g. the
    ErpCore2021_<task> folders over ErpCore2021/raw (exFAT has no symlinks)."""
    global _RAW_DIR
    import sys
    import moabb.datasets
    ds = getattr(moabb.datasets, class_name)(**kwargs)
    # Deliberate shortcut: module-global redirect of MOABB's storage-path lookup. Its MNE_DATASETS_<SIGN>_PATH
    # keys use a per-module sign (e.g. every BNCI dataset shares "BNCI"), not ds.code, and
    # several modules import get_dataset_path by name -- so patch every copy. One raw dir at
    # a time: fine for cache_dataset.py's per-dataset loop, not for two datasets in one process
    # concurrently.
    _RAW_DIR = os.path.abspath(os.path.join(dataset_path, raw_root or '', 'raw'))
    for name, mod in list(sys.modules.items()):
        if name.startswith('moabb.datasets') and hasattr(mod, 'get_dataset_path'):
            mod.get_dataset_path = _get_dataset_path
    return ds


def subject_runs(ds, subject: int):
    """[(session, run, mne.io.Raw)] for one subject, downloading it if needed."""
    # Deliberate shortcut: private _get_single_subject_data instead of get_data -- get_data's session
    # filter drops Lee2019's first session (selected_sessions (1, 2) vs session keys
    # '0'/'1', MOABB 1.5.0), and it also skips MOABB's cache layer we don't need.
    # Switch to get_data if a MOABB upgrade removes the private method.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sessions = ds._get_single_subject_data(subject)
    return [(s, r, raw) for s, runs in sessions.items() for r, raw in runs.items()]


class MoabbLoader(BaseSubjectLoader):
    """
    Cuts one trial per labelled event. Default: anchor = event + dataset.interval[0] (the
    MI/ERP onset MOABB defines), window = [anchor - pre_event_seconds, anchor +
    post_event_seconds] (post falls back to the interval length when unset). A dataset
    whose trials don't fit the global pre/post (P300 flashes, back-to-back PhysionetMI
    trials) sets metadata moabb.window = [t0, t1]: seconds relative to the raw event,
    used instead. moabb.onset_window = [t0, t1] is the same relative to the MOABB onset
    (event + interval[0], e.g. the MI cue), for datasets whose raw event is a trial start
    (EEG-FM-Compass cuts BNCI2014001 / 004 as [cue, cue + 4 / 4.5 s]). moabb.continuous = true (pretrain-only datasets whose trials overlap,
    e.g. P300 flashes ~0.1 s apart) ignores events and cuts every run into non-overlapping
    window_size_seconds windows with label 0. Channels are picked by name (metadata
    'original_label'); labels come from metadata targets' 'moabb_event'.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.moabb_subject = int(self._require_subject(subject_id)['moabb_subject'])
        m = self.data_metadata['moabb']
        self.ds = moabb_dataset(m['class'], self.data_root, m.get('raw_root'), **m.get('kwargs', {}))
        ch = self.data_metadata['channels']
        self.pick_names = [ch[str(i + 1)]['original_label'] for i in desired_channel_indices]
        t = self.data_metadata['targets']
        self.event_to_label = {t[k]['moabb_event']: int(k) for k in t if k.isdigit()}
        self.window = m.get('window')
        self.onset_window = m.get('onset_window')
        self.continuous = m.get('continuous', False)

    def _load_data(self):
        import mne
        lo, hi = self.ds.interval
        trials, labels, ranges, sessions = [], [], [], []
        runs = subject_runs(self.ds, self.moabb_subject)
        session_idx = {s: i for i, s in enumerate(dict.fromkeys(s for s, _, _ in runs))}
        tf = self.continuous_transform
        self.prefiltered = tf is not None
        for session, _, raw in runs:
            self._resample_if_needed(raw)
            data = raw.get_data(picks=self.pick_names).astype(np.float32)
            if (self.continuous or self.continuous_seconds) and len(segs := _zero_gap_segments(data, raw.info['sfreq'])) > 1:
                # MOABB stitched epochs with zero buffers (Weibo2014, Wang2016, Cho2017, ...): each piece is its own
                # recording, filtered alone and cut alone, never joined across a gap (user, 2026-10-03)
                for a, b in segs:
                    piece, sf_p = data[:, a:b], raw.info['sfreq']
                    if tf is not None:
                        piece, sf_p = tf(piece).numpy(), tf.sample_freq
                    win = int(round((self.continuous_seconds or self.standard_window) * sf_p))
                    for w, ve in cut_windows(piece, win, keep_short=True):
                        trials.append(w); labels.append(0); ranges.append((0, ve)); sessions.append(session_idx[session])
                continue
            scale = 1.0
            if tf is not None:     # bandpass + resample the whole run, then cut at the compiled rate
                data = tf(data).numpy()
                scale = tf.sample_freq / raw.info['sfreq']
            sf = raw.info['sfreq'] * scale
            if self.continuous or self.continuous_seconds:
                win = int(round((self.continuous_seconds or self.standard_window) * sf))
                for w, ve in cut_windows(data, win, min_real=1.0):       # whole windows only: the leftover is dropped
                    trials.append(w); labels.append(0); ranges.append((0, ve)); sessions.append(session_idx[session])
                continue
            if self.window:
                shift, pre, post = 0, int(round(-self.window[0] * sf)), int(round(self.window[1] * sf))
            elif self.onset_window:
                shift = int(round(lo * sf))
                pre, post = int(round(-self.onset_window[0] * sf)), int(round(self.onset_window[1] * sf))
            else:
                shift = int(round(lo * sf))
                pre = int(round(self.pre_event_seconds * sf))
                post = int(round((self.post_event_seconds or (hi - lo)) * sf))
            # Raw straight from _get_single_subject_data: events are on the stim channel
            # (MOABB's get_data pipeline is what would turn them into annotations).
            if mne.pick_types(raw.info, stim=True).size:
                events = mne.find_events(raw, shortest_event=0, verbose=False)
            else:
                events, _ = mne.events_from_annotations(raw, event_id=self.ds.event_id, verbose=False)
            code_to_label = {self.ds.event_id[k]: v for k, v in self.event_to_label.items()}
            for sample, _, code in events:
                if code not in code_to_label:
                    continue
                anchor = int(round((sample - raw.first_samp) * scale)) + shift
                window, vs, ve = cut_event_window(data, anchor, pre, post)
                if ve <= vs:
                    continue
                trials.append(window)
                labels.append(code_to_label[code])
                ranges.append((vs, ve))
                sessions.append(session_idx[session])
        if not trials:
            return None, None
        self._last_valid_ranges, self._last_sessions = ranges, sessions
        return np.stack(trials), np.array(labels, dtype=np.int64)


def _zero_gap_segments(data, sfreq, min_gap_s=0.1, min_seg_s=1.0):
    """[(start, end)] of the stretches between all-channel exact-zero gaps of at least min_gap_s (MOABB's buffers
    between stitched epochs); pieces shorter than min_seg_s are dropped. No gap -> one segment, the whole run."""
    zero = ~np.any(data != 0, axis=0)
    edges = np.flatnonzero(np.diff(np.r_[0, zero.astype(np.int8), 0]))
    gaps = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_gap_s * sfreq]
    cuts = [0] + [x for g in gaps for x in g] + [data.shape[1]]
    return [(a, b) for a, b in zip(cuts[::2], cuts[1::2]) if b - a >= min_seg_s * sfreq]


# --- BIDS-backed loading (NEMAR / OpenNeuro pretraining sets) ---------------------------------
# datas/<Name>/loader.py is `class Loader(BidsLoader)` and gen_metadata.py calls write_bids_metadata() (IO/metadata.py). Raw files stay in
# datas/<Name>/raw/ in BIDS layout (sub-*/[ses-*/]eeg/*_eeg.<edf|bdf|set|vhdr>). Pretraining only: every recording is
# cut into continuous windows, events ignored (labels 0).

BIDS_EXT = ('.edf', '.bdf', '.set', '.vhdr')
# channels that are not scalp EEG even when the file types them EEG (BIDS channels.tsv types are often n/a)
_NON_EEG = re.compile(r'(EOG|ECG|EKG|EMG|ACC|GSR|RESP|TRIG|STATUS|STI|PHOTO|AUDIO|SPO2|PULSE|^X\d|^Y\d|^Z\d)', re.I)


def _bids_runs(root, task=None):
    """{bids subject label: [raw file paths relative to root]} under root/raw, optional task filter."""
    import glob
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, 'raw', 'sub-*', '**', 'eeg', '*_eeg.*'), recursive=True)):
        if not f.endswith(BIDS_EXT) or (task and f'_task-{task}_' not in os.path.basename(f)):
            continue
        sub = os.path.basename(f).split('_')[0][4:]
        runs.setdefault(sub, []).append(os.path.relpath(f, root))
    return runs


def _read_raw(path, preload=False):
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return mne.io.read_raw(path, preload=preload, verbose=False)


def _eeg_names(raw):
    """Scalp EEG channels: typed EEG, not named like an auxiliary channel, and a 10-05 site MNE knows (consumer
    headsets type their quality / battery / timestamp columns EEG too; mastoid leads named 'Mastoid L' drop out)."""
    import mne
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)
            if not _NON_EEG.search(raw.ch_names[i]) and get_standard_coords(raw.ch_names[i]) is not None]


class BidsLoader(BaseSubjectLoader):
    """Every recording of the subject: scalp EEG picked by name (missing ones zero), resampled to the metadata rate,
    split at EEGLAB 'boundary' annotations and all-channel zero gaps (never joined across a discontinuity), each
    piece filtered alone (continuous_transform) and cut into continuous_seconds windows (tail kept if >= half a
    window, zero-padded, marked by valid_ranges). Session index = recording order of the ses-* folders."""
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        if not self.continuous_seconds:
            raise ValueError("BidsLoader is pretraining-only: set compile_params.continuous_seconds")
        self.files = [self._resolve(r['file']) for r in self._require_subject(subject_id)['runs']]
        ch = self.data_metadata['channels']
        self.pick_names = [ch[str(i + 1)]['original_label'] for i in desired_channel_indices]

    def _prepare(self, raw):
        """Hook: a dataset whose channel names differ from its metadata labels renames them here."""
        return raw

    def _load_data(self):
        trials, ranges, sessions, ses_idx = [], [], [], {}
        for path in self._existing(self.files):
            try:
                raw = self._prepare(_read_raw(path, preload=True))
            except Exception as e:
                print(f"  [Warning] {path}: {e}")
                continue
            self._resample_if_needed(raw)
            names = {c.upper(): c for c in raw.ch_names}                 # case-insensitive (Pz / PZ across labs)
            rows = [i for i, c in enumerate(self.pick_names) if c.upper() in names]
            if not rows:
                continue
            data = np.zeros((len(self.pick_names), raw.n_times), dtype=np.float32)
            data[rows] = np.nan_to_num(raw.get_data(picks=[names[self.pick_names[i].upper()] for i in rows]))
            cuts = sorted({0, raw.n_times} | {int(round((a['onset'] - raw.first_time) * raw.info['sfreq']))
                                                for a in raw.annotations if a['description'].lower() == 'boundary'})
            ses = next((p for p in path.split(os.sep) if p.startswith('ses-')), '')
            si = ses_idx.setdefault(ses, len(ses_idx))
            for a, b in zip(cuts[:-1], cuts[1:]):
                for g0, g1 in _zero_gap_segments(data[:, a:b], raw.info['sfreq']):
                    piece, sf = self._filter_run(data[:, a + g0:a + g1])
                    for w, ve in cut_windows(piece, int(round(self.continuous_seconds * sf)), keep_short=True):
                        trials.append(w); ranges.append((0, ve)); sessions.append(si)
        if not trials:
            return None, None
        self._last_valid_ranges, self._last_sessions = ranges, sessions
        return np.stack(trials), np.zeros(len(trials), dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# TUH EEG corpus and its annotated subsets (datas/pretrain/TUEG, datas/finetune/TUAB, TUEV, TUSL, TUAR). The recordings are
# clinical EDF files whose channel names carry a prefix and a reference suffix that vary by montage file ('EEG FP1-REF',
# 'EEG FP1-LE'), at 250 / 256 / 400 / 512 Hz.
# ---------------------------------------------------------------------------------------------------------------------
TUH_NAME = re.compile(r'^EEG\s+([A-Z0-9]+)-(REF|LE|AR\w*|AVG\w*)$', re.I)
# the 21 scalp sites of the 10-20 system every TUH montage has (TUAB's 21 channels; old names T3/T4/T5/T6 resolve to
# T7/T8/P7/P8 in IO/dataset.py)
SITES21 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
           'A1', 'A2']


def rename_channels(raw):
    """'EEG FP1-REF' -> 'FP1' (first occurrence of a site wins); everything else keeps its name."""
    new = {}
    for c in raw.ch_names:
        m = TUH_NAME.match(c)
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
    return os.path.basename(path).split('_')[0]
