import os
import json
import random
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Callable, Any

from IO.channels import NON_EEG_CHANNELS, channel_xyz, get_standard_coords, idw_matrix, resolve_target_channels
from IO.preprocessing import build_normalizer_from_config, cache_suffix, window_continuous_signal
from IO.checks import sanity_check_base, sanity_check_wrapper
from IO.masking import build_masking_strategy_from_config
from IO.wrappers import FinetuneDataset, PretrainDataset

# --- Base Dataset ---

# pretrain: a channel with std below this x the subject's median channel std is treated as padding. 0.10 catches
# dead electrodes and the recording reference (e.g. Cz in the Tsinghua SSVEP sets, mastoids): 309 subject-channels
# over the corpus on 2026-09-26, versus 88 at 0.05 (only the near-zero ones).
FLAT_RATIO = 0.10


class EEGDataset(Dataset):
    """
    Loads EEG data from multiple subjects/datasets into a unified tensor.
    When assemble_trials=True, flattens each subject's trials into a continuous
    signal and cuts non-overlapping windows of assembly_params['window_length'].
    """
    def __init__(
        self,
        config: Dict,
        loading_tasks: List[Dict[str, Any]],
        desired_channels: List[str],
        assemble_trials: bool = False,
        assembly_params: Optional[Dict] = None
    ):
        self.config = config
        self.channel_names = desired_channels
        self.Nc = len(desired_channels)
        # Channel layout (docs/notes/decisions/0023-real-channel-coordinates.md): every EEG channel of the dataset is kept, canonical-name channels on their
        # slot, the others in free slots; > Nc channels are reduced to the canonical sites. Coordinates: MNE's template
        # position for each channel name (channel_xyz).
        self._plans = {}
        self.assemble_trials = assemble_trials
        self.assembly_params = assembly_params or {}

        all_data_chunks: List[torch.Tensor] = []
        all_label_chunks: List[torch.Tensor] = []
        all_subject_chunks: List[torch.Tensor] = []
        all_dataset_names: List[str] = []
        all_coords: List[torch.Tensor] = []
        all_valid_channels: List[torch.Tensor] = []  # per-task [Nc] bool, True = real (not zero-padded) channel
        all_named_slots: List[torch.Tensor] = []     # per-task [Nc] bool, True = the slot holds its canonical-name channel
        all_valid_length: List[int] = []             # per-task original T, before cross-subject max_T padding
        all_row_valid_start: List[torch.Tensor] = []  # per-task [N_rows] real-content start, per row
        all_row_valid_end: List[torch.Tensor] = []    # per-task [N_rows] real-content end, per row

        print(f"Loading {len(loading_tasks)} subject-dataset tasks... (assembly={'on' if assemble_trials else 'off'})")
        for task in loading_tasks:
            try:
                result = self._load_task(task, desired_channels)
            except Exception as e:
                print(f"Failed to load Subject {task.get('subject_id')} ({task.get('dataset_name')}): {e}")
                continue
            if result is None:
                continue

            all_data_chunks.append(result['data'])
            all_label_chunks.append(result['labels'])
            all_subject_chunks.append(torch.full((len(result['data']),), int(result['subject_id']), dtype=torch.long))
            all_dataset_names.extend([result['dataset_name']] * len(result['data']))
            all_coords.append(result['coords'])
            all_valid_channels.append(result['valid_channels'])
            all_named_slots.append(result['named_slots'])
            all_valid_length.append(result['valid_length'])
            all_row_valid_start.append(result['row_valid_start'])
            all_row_valid_end.append(result['row_valid_end'])

        if not all_data_chunks:
            raise RuntimeError("No datasets were loaded successfully.")

        # Standardize temporal length across all subjects
        max_T = max(d.shape[-1] for d in all_data_chunks)
        standardized = []
        for d in all_data_chunks:
            if d.shape[-1] < max_T:
                pad = torch.zeros((d.shape[0], d.shape[1], max_T - d.shape[-1]), dtype=d.dtype)
                d = torch.cat([d, pad], dim=-1)
            standardized.append(d)

        self.data = torch.cat(standardized)
        self.labels = torch.cat(all_label_chunks)
        self.subject_data = torch.cat(all_subject_chunks)
        self.dataset_names = all_dataset_names
        self.all_coords = all_coords
        self.all_valid_channels = all_valid_channels
        # Which valid slots hold the channel their canonical name says (a non-canonical channel sits in a free slot). Name-based logic (channel subsampler, backbone_eval's motor-3 ->
        # bci-22 test) reads only these.
        self.all_named_slots = all_named_slots
        self.all_valid_length = all_valid_length
        # Per-ROW (not per-task, unlike all_valid_length above): indexed the same way
        # self.labels/self.data rows are, since real-content bounds vary WITHIN a task
        # (an assembled window's tail pad, or an event-anchored trial's own pre/post pad
        # near a recording edge). data[:, :, row_valid_start[i]:row_valid_end[i]] is real
        # content for row i; everything outside is zero pad. Used by PretrainDataset to
        # keep masking from picking patches out of padded content.
        self.row_valid_start = torch.cat(all_row_valid_start)
        self.row_valid_end = torch.cat(all_row_valid_end)

        self.trial_to_coords_idx = []
        for i, d in enumerate(standardized):
            self.trial_to_coords_idx.extend([i] * d.shape[0])

        print(f"Loaded {len(self.data)} trials, standardized to length {max_T}. Shape: {tuple(self.data.shape)}")

    def _load_task(self, task: Dict[str, Any], desired_channels: List[str]) -> Optional[Dict[str, Any]]:
        """
        Loads one subject-dataset task, channel-pads it into self.Nc unified
        channels, applies the preprocessing transform, and (for pretrain)
        windows it into fixed-length trials. Returns None if the subject has
        no data to load.
        """
        ds_name = task['dataset_name']
        subject_id = task['subject_id']
        transform = task['transform']
        ds_config = task['dataset_config']

        if ds_name not in self._plans:
            self._plans[ds_name] = self._channel_plan(desired_channels, ds_config)
        plan = self._plans[ds_name]
        ds_indices, target_pos = plan['src'], plan['slots']

        # Train-time read: compiled cache only (see cache_dataset.py) — dataset-specific
        # loading code (datas/<Name>/loader.py) never runs at train time. The cached
        # array holds ALL native channels in metadata.json's index order (compile time
        # keeps every channel, no target-channel subsetting), so ds_indices indexes
        # directly into it, no extra mapping needed.
        dataset_path = ds_config['dataset_params']['dataset_path']
        cache_path = os.path.join(dataset_path, 'cache', f"{subject_id}_{task['cache_suffix']}.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No compiled cache at {cache_path}. Run "
                f"`python cache_dataset.py --config configs/compile.json` first "
                f"(and make sure compile.json's sample_freq/bandpass_filter match "
                f"this config's preprocess_params)."
            )
        npz = np.load(cache_path)
        data_np = npz['data']
        # dataset_params.<mode>.<ds>.trial_fraction (analysis panels, whole trials only): keep a fixed random share of the
        # subject's real trials, so a panel that needs ~1000 trials of a dataset keeps every subject without loading them
        # whole. Picked here, before the transform and the 64-slot padding, which would otherwise run on every trial.
        frac = ds_config['dataset_params'].get('trial_fraction', 1.0)
        keep = None
        if not self.assemble_trials and frac < 1.0 and len(data_np):
            keep = np.sort(np.random.default_rng(zlib.crc32(f"{ds_name}/{subject_id}".encode()))
                           .permutation(len(data_np))[:max(1, int(round(frac * len(data_np))))])
            data_np = data_np[keep]
        data_np = data_np[:, ds_indices, :]
        if data_np.shape[0] == 0:
            return None
        coords_np = plan['coords']
        src_names = plan['labels']

        raw_data = torch.from_numpy(data_np.astype(np.float32))  # (N, C, T)
        N, _, T = raw_data.shape
        # Pretrain only: a channel far flatter than the subject's others (std < FLAT_RATIO x the
        # median channel std, measured BEFORE the per-trial z-score, which would blow its noise
        # up to unit variance) is a dead electrode or the reference -- treated as padding below.
        flat = torch.zeros(len(ds_indices), dtype=torch.bool)
        if self.assemble_trials:
            ch_std = raw_data.transpose(0, 1).reshape(len(ds_indices), -1).std(dim=1)
            flat = ch_std < FLAT_RATIO * ch_std.median()
            if flat.any():
                print(f"  [{ds_name} S{subject_id}] flat channels -> padding: "
                      f"{[src_names[i] for i in flat.nonzero().flatten().tolist()]}")
        named = plan['named']
        if plan['interp']:
            # > Nc channels: every canonical site, copied where the recording has it (and it is not flat),
            # IDW-interpolated from its good electrodes where not. Dead electrodes are left out of the interpolation.
            W, coords_np = self._interp_matrix(plan, ~flat.numpy(), desired_channels)
            raw_data = torch.einsum('sc,nct->nst', torch.from_numpy(W).float(), raw_data)
            target_pos, named = list(range(self.Nc)), [True] * self.Nc
            flat = torch.zeros(self.Nc, dtype=torch.bool)
        # Per-trial real-content bounds, compiled-rate samples (see cache_dataset.py /
        # IO/loader.py's get_subject_data) -- 'valid_start'/'valid_end' absent (an older
        # cache from before this existed) means "every trial fully real", same default
        # get_subject_data itself uses.
        cache_valid_start = npz['valid_start'].tolist() if 'valid_start' in npz else [0] * N
        cache_valid_end = npz['valid_end'].tolist() if 'valid_end' in npz else [T] * N
        if keep is not None and 'valid_start' in npz:
            cache_valid_start, cache_valid_end = [cache_valid_start[i] for i in keep], [cache_valid_end[i] for i in keep]

        # Transform (bandpass/resample/normalize) on the REAL channels only, before padding —
        # normalizing after zero-padding folds the zero-filled missing-channel rows into the
        # trial's mean/std (see IO/preprocessing.py's per-trial zscore/robust normalize), which
        # skews scale differently per dataset depending how many channels it's missing relative
        # to canonical_channels (e.g. BNCI2014001 is ~2/3 zero-padded channels — that's a much
        # bigger normalization bias than a near-complete dataset like PhysionetMI).
        if transform is not None:
            # Whole (N, C, T) batch in one call -- Normalizer._normalize reduces
            # per-trial (its own mean/std/median, never pooled across N), so this is
            # numerically identical to the old torch.stack([transform(raw_data[i])
            # for i in range(N)]) loop, just without holding N separately-normalized
            # tensors in a Python list before the stack copies them into one buffer
            # (see IO/preprocessing.py's Normalizer docstring).
            raw_data = transform(raw_data)
        post_transform_T = raw_data.shape[-1]  # real (non-padded) length for finetune's per-trial mask

        padded = torch.zeros((N, self.Nc, post_transform_T), dtype=torch.float32)
        padded[:, target_pos, :] = raw_data

        if self.assemble_trials:
            target_L = self.assembly_params.get('window_length', padded.shape[-1])
            padded, labels, row_valid_start, row_valid_end = window_continuous_signal(
                padded, target_L, ds_name, subject_id,
                valid_ranges=list(zip(cache_valid_start, cache_valid_end)),
                min_real_fraction=self.assembly_params.get('window_min_real', 0.5))
            # preprocess_params.window_fraction (pretrain only): keep this fraction of the
            # subject's windows -- the first n_keep of one fixed permutation seeded by
            # (window_fraction_seed, dataset, subject), so every subject stays in the corpus
            # and, at one seed, a smaller fraction is a subset of a larger one (corpus
            # sizes tiny 5% < small 20% < medium 50% < large 100%, 2026-09-24). A dataset's own
            # dataset_params.pretrain.<ds>.window_fraction multiplies in: the corpus builder balances
            # paradigms by capping windows per subject, never by dropping subjects -- the product
            # keeps the nesting across corpus sizes.
            # dataset_params.pretrain.<ds>.max_windows_per_subject (the corpus builder's per-subject time cap) first restricts the
            # subject to the first that many windows of the same permutation; the fractions then thin what is left.
            frac = self.assembly_params.get('window_fraction', 1.0) * ds_config['dataset_params'].get('window_fraction', 1.0)
            max_windows = ds_config['dataset_params'].get('max_windows_per_subject')
            if (frac < 1.0 or max_windows) and len(padded):
                seed = zlib.crc32(f"{self.assembly_params.get('window_fraction_seed', 0)}/{ds_name}/{subject_id}".encode())
                order = np.random.default_rng(seed).permutation(len(padded))
                if max_windows:
                    order = order[:max_windows]
                n_keep = max(1, int(round(frac * len(order))))
                keep = torch.from_numpy(np.sort(order[:n_keep]))
                padded, labels = padded[keep], labels[keep]
                row_valid_start, row_valid_end = row_valid_start[keep], row_valid_end[keep]
        else:
            labels = torch.from_numpy(npz['labels'].astype(np.int64) if keep is None else npz['labels'][keep].astype(np.int64))
            # Every row here is one real, untouched-by-assembly trial -- use the cache's
            # own per-trial bounds directly (cross-subject max_T padding, applied later in
            # EEGDataset.__init__, is handled separately via all_valid_length/FinetuneDataset).
            row_valid_start = torch.tensor(cache_valid_start, dtype=torch.long)
            row_valid_end = torch.tensor(cache_valid_end, dtype=torch.long)

        task_coords = torch.zeros((self.Nc, 3), dtype=torch.float32)
        task_coords[target_pos] = torch.from_numpy(coords_np.astype(np.float32))

        valid_channels = torch.zeros(self.Nc, dtype=torch.bool)
        valid_channels[target_pos] = True
        if flat.any():
            dead = [target_pos[i] for i in flat.nonzero().flatten().tolist()]
            valid_channels[dead] = False
            padded[:, dead] = 0
            task_coords[dead] = 0
        named_slots = torch.zeros(self.Nc, dtype=torch.bool)
        named_slots[[p for p, n in zip(target_pos, named) if n]] = True
        named_slots &= valid_channels

        return {
            'data': padded,
            'labels': labels,
            'dataset_name': ds_name,
            'subject_id': subject_id,
            'coords': task_coords,
            'valid_channels': valid_channels,
            'named_slots': named_slots,
            'valid_length': post_transform_T,
            'row_valid_start': row_valid_start,
            'row_valid_end': row_valid_end,
        }

    # Old → canonical label aliases (covers both 10-20 naming conventions)
    _LABEL_ALIASES = {
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8',
        'A1': 'TP9', 'A2': 'TP10',
        # BCICIV_1-style intermediate-ring labels -> nearest canonical 10-10 site
        'CFC1': 'FC1', 'CFC2': 'FC2', 'CFC3': 'FC3', 'CFC4': 'FC4',
        'CFC5': 'FC5', 'CFC6': 'FC6', 'CFC7': 'FT7', 'CFC8': 'FT8',
        'CCP1': 'CP1', 'CCP2': 'CP2', 'CCP3': 'CP3', 'CCP4': 'CP4',
        'CCP5': 'CP5', 'CCP6': 'CP6', 'CCP7': 'TP7', 'CCP8': 'TP8',
        'PO1': 'PO3', 'PO2': 'PO4',
    }

    def _normalize_label(self, label: str) -> str:
        up = label.strip().upper()
        return self._LABEL_ALIASES.get(up, up)

    def _channel_plan(self, desired_channels: List[str], ds_config: Dict) -> Dict:
        """Which cached channels to read and where they go. A channel whose label (with the 10-20 aliases) is a
        canonical name keeps that slot; every other EEG channel with a known position fills a free slot. More than Nc channels -> 'interp': all of them feed
        _interp_matrix. -> {src, slots, labels, coords, named, interp, pos}."""
        chans = ds_config['data_metadata']['channels']
        include_non_eeg = ds_config['dataset_params'].get('include_non_eeg_channels', False)
        keys = sorted((k for k in chans if isinstance(k, str) and k.isdigit()), key=int)
        slot_of = {self._normalize_label(n): s for s, n in enumerate(desired_channels)}
        src, labels, pos, slot, dropped = [], [], [], [], []
        for k in keys:
            info = chans[k]
            label = info.get('label', '') if isinstance(info, dict) else ''
            s = slot_of.get(self._normalize_label(label))
            if s is not None and s in slot:
                s = None                                       # two labels for one site (e.g. T3 and T7)
            if s is None and not include_non_eeg and label.upper() in NON_EEG_CHANNELS:
                continue
            p = channel_xyz(info) if isinstance(info, dict) else None
            if p is None:
                dropped.append(label)
                continue
            src.append(int(k) - 1); labels.append(label); pos.append(p); slot.append(s)
        if dropped:
            print(f"  [channel map] no position, dropped: {dropped}")
        # canonical-slot order first, the other channels after in metadata order (the per-trial z-score sums channels
        # in read order)
        order = sorted(range(len(src)), key=lambda j: (slot[j] is None, slot[j] if slot[j] is not None else j))
        src, labels, pos, slot = ([v[j] for j in order] for v in (src, labels, pos, slot))
        named = [s is not None for s in slot]
        pos = np.array(pos)
        interp = len(src) > self.Nc
        if not interp:
            free = iter(s for s in range(self.Nc) if s not in slot)
            slot = [s if s is not None else next(free) for s in slot]
        print(f"  [channel map real] {len(src)} EEG channels, {sum(named)} on canonical slots"
              + (f", > {self.Nc}: interpolated to the canonical sites" if interp else ""))
        return {'src': src, 'slots': slot, 'labels': labels, 'coords': pos.astype(np.float32), 'named': named,
                'interp': interp, 'pos': pos}

    def _interp_matrix(self, plan: Dict, good: np.ndarray, desired_channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """> Nc channels -> (W [Nc, n_src] mapping the recording to the Nc canonical sites, coords [Nc, 3]). A site
        the recording has (and whose electrode is good) is copied (one-hot row, the electrode's own position); every
        other site is an inverse-distance-weighted row (idw_matrix) from the good electrodes, at the site's standard
        position."""
        n = len(plan['src'])
        W = np.zeros((self.Nc, n))
        coords = np.zeros((self.Nc, 3))
        have = {s: j for j, s in enumerate(plan['slots']) if s is not None and good[j]}
        missing = [s for s in range(self.Nc) if s not in have]
        for s, j in have.items():
            W[s, j], coords[s] = 1.0, plan['pos'][j]
        if missing:
            std = np.array([get_standard_coords(desired_channels[s]) for s in missing], dtype=np.float64)
            gi = np.flatnonzero(good)
            W[np.ix_(missing, gi)] = idw_matrix(plan['pos'][gi], std)
            coords[missing] = std
        return W, coords.astype(np.float32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# --- Sanity checks ---

# --- Dataset Wrappers ---

# --- Factory ---

def build_dataset_from_config(config_dict: Dict, transform: Optional[Callable] = None, mode: str = 'pretrain',
                               assemble_trials: Optional[bool] = None) -> Dataset:
    ds_mode        = mode if mode in ('pretrain', 'finetune') else 'pretrain'
    dataset_params = config_dict.get('dataset_params', {}).get(ds_mode, {})
    pp             = config_dict.get('preprocess_params', {})
    patch_len      = pp.get('patch_length', 100)
    patch_stride   = pp.get('patch_stride', patch_len)

    loading_tasks = []
    for ds_name, ds_args in dataset_params.items():
        meta_path = os.path.join(ds_args['dataset_path'], 'metadata.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        data_metadata = metadata.get('data_metadata', {})
        data_structure = metadata.get('data_structure', {})

        ds_transform = transform if transform is not None else build_normalizer_from_config(config_dict)
        ds_cache_suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                                        pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0),
                                        pp.get('continuous_seconds', 0.0) if ds_mode == 'pretrain' else 0.0)

        loader_config = {
            'dataset_params': ds_args,
            'data_metadata': data_metadata,
            'data_structure': data_structure
        }

        requested_subjects = ds_args['subject_to_use']
        if requested_subjects in (['all'], 'all'):
            all_ids = list(data_structure.keys())
            try:
                requested_subjects = sorted(all_ids, key=int)
            except ValueError:
                requested_subjects = sorted(all_ids)

        for sub_id in requested_subjects:
            loading_tasks.append({
                'dataset_name': ds_name,
                'subject_id': sub_id,
                'transform': ds_transform,
                'dataset_config': loader_config,
                'cache_suffix': ds_cache_suffix,
            })

    target_channels = resolve_target_channels(dataset_params, pp=pp)

    if assemble_trials is None:
        assemble_trials = mode == 'pretrain'  # explicit override: e.g. real
        # per-trial labels for codebook diagnostics (pretrain/tokenizer normally assemble
        # trials into continuous-signal windows, which discards real labels -- see
        # IO/preprocessing.py's window_continuous_signal)
    assembly_params = pp

    base_dataset = EEGDataset(
        config=config_dict,
        loading_tasks=loading_tasks,
        desired_channels=target_channels,
        assemble_trials=assemble_trials,
        assembly_params=assembly_params
    )
    sanity_check_base(base_dataset)

    if mode == 'base':
        return base_dataset
    elif mode == 'pretrain':
        # The tokenizer phase ignores masks; train_pretrain.py re-applies the strategy (with
        # its epoch) from the first masked epoch on.
        strategy = build_masking_strategy_from_config(pp.get('mask', {}))
        ds = PretrainDataset(base_dataset, patch_len=patch_len, patch_stride=patch_stride,
                             masking_strategy=strategy)
        sanity_check_wrapper(ds)
        return ds
    elif mode == 'finetune':
        ds = FinetuneDataset(base_dataset)
        sanity_check_wrapper(ds)
        return ds
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected one of: base, pretrain, finetune.")
