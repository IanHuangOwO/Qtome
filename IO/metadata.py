"""Dataset setup tools, run once when a dataset is added (not at training or compile time): the metadata.json writers that
datas/<Name>/gen_metadata.py calls (MOABB-backed: write_moabb_metadata; BIDS-backed: write_bids_metadata). The loaders they
describe are in IO/loader.py; downloads are tools/misc/fetch_datasets.py."""
import json
import os
from typing import Dict, List

from IO.loader import _bids_runs, _eeg_names, _read_raw, moabb_dataset, subject_runs


def write_moabb_metadata(root: str, name: str, class_name: str, dataset_info: Dict,
                         target_labels: Dict[str, str], kwargs: Dict = None,
                         window: List[float] = None, continuous_seconds: float = None,
                         onset_window: List[float] = None, cohort: str = None, postprocess=None,
                         raw_root: str = None) -> Dict:
    """
    Writes <root>/metadata.json from MOABB: EEG channel names and sample rate from the
    first subject's first run (downloads it if needed), subjects from ds.subject_list,
    targets = MOABB event names. target_labels maps event name -> readable label; its key
    order sets the label indices (so a migrated dataset can keep its old label order),
    events it leaves out follow in sorted order. postprocess(meta) may add to the dict before
    it is written (e.g. the dataset's own electrode coordinates).
    """
    import mne
    kwargs = kwargs or {}
    ds = moabb_dataset(class_name, root, raw_root, **kwargs)
    raw = subject_runs(ds, ds.subject_list[0])[0][2]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    events = [e for e in target_labels if e in ds.event_id] + sorted(set(ds.event_id) - set(target_labels))
    lo, hi = ds.interval
    # Where the MOABB event sits inside each compiled trial, in seconds from trial start --
    # read by panels.shared.event_onset.lookup_event_onset_sample for the event line on time-axis
    # plots. window=[t0, t1] is relative to the raw event (-> -t0); the default window is
    # centred on event + interval[0] (the MI/stimulus onset MOABB defines) with the global
    # pre_event_seconds before it; onset_window=[t0, t1] is relative to that onset (-> -t0);
    # continuous windows have no event.
    if continuous_seconds:
        onset = None
    elif window or onset_window:
        onset = 0.0 - (window or onset_window)[0]  # 0.0 - x, not -x: no '-0.0' in metadata
    else:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'compile.json')) as f:
            onset = json.load(f)['compile_params'].get('pre_event_seconds', 0.0)
    meta = {
        "data_metadata": {
            "dataset_name": name,
            "dataset_info": dataset_info,
            **({"event_onset_seconds": onset} if onset is not None else {}),
            "moabb": {"class": class_name, "kwargs": kwargs, "code": ds.code,
                      **({"raw_root": raw_root} if raw_root else {}),
                      **({"window": window} if window else {}),
                      **({"onset_window": onset_window} if onset_window else {}),
                      **({"continuous": True} if continuous_seconds else {})},
            "acquisition": {
                "sample_frequency": raw.info['sfreq'],
                "window_size_seconds": continuous_seconds or ((w := window or onset_window)[1] - w[0] if (window or onset_window) else hi - lo),
                "num_subjects": len(ds.subject_list),
            },
            "targets": {"count": 1, "type": "pretrain_dummy",
                        "0": {"label": "dummy (continuous window, no task label)", "moabb_event": None}}
            if continuous_seconds else {"count": len(events), "type": ds.paradigm,
                        **{str(i): {"label": target_labels.get(e, e), "moabb_event": e}
                           for i, e in enumerate(events)}},
            "channels": {"count": len(eeg), **{str(i + 1): {"label": n, "original_label": n}
                                               for i, n in enumerate(eeg)}},
            **({"cohort": cohort} if cohort else {}),   # same people as another dataset (split_pretrain_subjects)
        },
        "data_structure": {str(s): {"moabb_subject": s} for s in ds.subject_list},
    }
    if postprocess:
        postprocess(meta)
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote {name}/metadata.json: {len(ds.subject_list)} subjects, {len(eeg)} EEG channels, "
          f"{raw.info['sfreq']} Hz, events {events}")
    return meta


def write_bids_metadata(root: str, name: str, dataset_info: Dict, task: str = None, cohort: str = None) -> Dict:
    """Writes <root>/metadata.json for a BIDS raw tree: channels = union of every subject's scalp EEG channels,
    matched case-insensitively (first-seen order; a subject missing one gets a zero row there, which IO/dataset.py
    treats as padding), sample rate = the most common one (others are resampled), subjects numbered by their BIDS
    label when every label is an integer (so datasets of one cohort share ids), else 1..N in sorted order (the label
    is kept as bids_subject)."""
    from collections import Counter
    runs = _bids_runs(root, task)
    channels, rates = [], Counter()
    for sub, files in runs.items():
        raw = _read_raw(os.path.join(root, files[0]))
        rates[raw.info['sfreq']] += 1
        seen = {c.upper() for c in channels}
        channels += [c for c in _eeg_names(raw) if c.upper() not in seen]     # labs differ in case (Pz / PZ)
    numeric = all(s.isdigit() for s in runs)
    ids = {s: (int(s) if numeric else i + 1) for i, s in enumerate(sorted(runs, key=lambda s: (len(s), s)))}
    meta = {
        "data_metadata": {
            "dataset_name": name, "dataset_info": dataset_info,
            "bids": {"task": task},
            "acquisition": {"sample_frequency": rates.most_common(1)[0][0], "window_size_seconds": None,
                            "num_subjects": len(runs)},
            "targets": {"count": 1, "type": "pretrain_dummy", "0": {"label": "dummy (continuous window, no task label)"}},
            "channels": {"count": len(channels), **{str(i + 1): {"label": c, "original_label": c}
                                                   for i, c in enumerate(channels)}},
            **({"cohort": cohort} if cohort else {}),
        },
        "data_structure": {str(ids[s]): {"bids_subject": s, "runs": [{"file": f} for f in files]}
                           for s, files in sorted(runs.items(), key=lambda kv: ids[kv[0]])},
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote {name}/metadata.json: {len(runs)} subjects, {sum(map(len, runs.values()))} recordings, "
          f"{len(channels)} EEG channels, rates {dict(rates)}")
    return meta
