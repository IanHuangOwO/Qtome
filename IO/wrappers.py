"""The training-time views of an EEGDataset: PretrainDataset (windows -> patches with masks), FinetuneDataset (trials), and the montage-batching sampler."""
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Optional, Any

from IO.channels import load_montage_channels
from IO.masking import MaskingStrategy, build_masking_strategy_from_config
from IO.preprocessing import num_patches, slice_patches, window_continuous_signal


def _resolve_default_patch_len(base_dataset: 'EEGDataset') -> int:
    model_type = base_dataset.config.get('training_params', {}).get('pretrain', {}).get('model_type', 'Qtome')
    preprocess = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
    return preprocess.get('patch_length', 200)


class PretrainDataset(Dataset):
    """
    Wraps EEGDataset for masked pretraining.
    Yields: (x_patches, coords, mask, time_indices, label, valid_channels)
      x_patches:      [C, P, L]
      coords:         [C, 3]
      mask:           [C * P] bool
      time_indices:   [P]
      label:          scalar
      valid_channels: [C] bool, True = real (not zero-padded) channel
    """
    def __init__(
        self,
        base_dataset: 'EEGDataset',
        patch_len: Optional[int] = None,
        patch_stride: Optional[int] = None,
        masking_strategy: Optional[MaskingStrategy] = None,
    ):
        self.base_dataset = base_dataset
        if patch_len is None:
            patch_len = _resolve_default_patch_len(base_dataset)
        self.patch_len = patch_len
        self.patch_stride = patch_stride or patch_len
        self.num_patches = num_patches(base_dataset.data.shape[-1], self.patch_len, self.patch_stride)

        # Per-trial (channel, patch) validity -- excludes zero-padded channels AND, for an
        # assembled window's zero tail (window_continuous_signal), patches that start past
        # the window's real content. Fixed after construction; every mask is drawn inside it.
        self._valid_masks = self._build_valid_masks()
        self.set_masking(masking_strategy or build_masking_strategy_from_config({}))

        n = len(base_dataset)
        print(f"\n--- PretrainDataset ---")
        print(f"  {n} trials | {self.num_patches} patches/trial (patch_len={self.patch_len}, "
              f"patch_stride={self.patch_stride}) | {self.masking_strategy.describe()}")
        print(f"----------------------------\n")

    def _build_valid_masks(self):
        """Per-trial [Nc*num_patches] bool -- True at (channel, patch) positions that are
        BOTH a real (not zero-padded) channel and a genuinely-real patch position (a
        patch whose start sample falls within [row_valid_start, row_valid_end) -- outside
        that, whether an assembled window's tail pad or an event-anchored trial's own
        leading/trailing pad near a recording edge, it's padding, see
        IO/preprocessing.py's window_continuous_signal / cut_event_window). Masks are never
        drawn outside it, so masking never spends its budget on content that's already
        known-zero."""
        bd = self.base_dataset
        patch_starts = torch.arange(self.num_patches) * self.patch_stride  # [P]
        masks = []
        for trial_idx in range(len(bd)):
            coords_idx = bd.trial_to_coords_idx[trial_idx]
            valid_channels = bd.all_valid_channels[coords_idx]           # [Nc] bool
            valid_patch = (patch_starts >= bd.row_valid_start[trial_idx]) \
                & (patch_starts < bd.row_valid_end[trial_idx])           # [P] bool
            masks.append((valid_channels.unsqueeze(1) & valid_patch.unsqueeze(0)).reshape(-1))
        return masks

    def set_masking(self, masking_strategy: MaskingStrategy):
        """(Re)draw every trial's mask ([C*N] bool) from the strategy's current epoch (the
        training loop calls this every masked epoch, see train_pretrain.py). Any DataLoader
        built on this dataset must be rebuilt afterwards (with persistent_workers, workers hold
        their own copy of the dataset from spawn time).

        masking_strategy.subsampler (ChannelSubsampler, optional): per trial, a sparse montage
        to keep; removed channels leave the valid set BEFORE the mask is drawn, and
        __getitem__ turns them into padding (zero signal, not valid, no loss)."""
        self.masking_strategy = masking_strategy
        subsampler = masking_strategy.subsampler
        bd = self.base_dataset
        norm = bd._normalize_label
        name_to_idx = {norm(n): i for i, n in enumerate(bd.channel_names)}
        to_idx = lambda montages: [[name_to_idx[norm(n)] for n in load_montage_channels(m) if norm(n) in name_to_idx]
                                   for m in montages]
        if subsampler is not None and not hasattr(self, '_montage_idx'):
            self._montage_idx = to_idx(subsampler.montages)
        for mode in masking_strategy.modes:                                 # MontageMask: the same channel lookup
            if hasattr(mode, 'montage_idx') and mode.montage_idx is None:
                mode.montage_idx = to_idx(mode.montages)
        self._keep, self._masks = [], []
        for i in range(len(bd)):
            valid, keep = self._valid_masks[i], None
            if subsampler is not None:
                ti = bd.trial_to_coords_idx[i]
                keep = subsampler.sample(bd.all_valid_channels[ti] & bd.all_named_slots[ti], self._montage_idx)
                if keep is not None:
                    valid = valid & keep.repeat_interleave(self.num_patches)
            self._keep.append(keep)
            coords = bd.all_coords[bd.trial_to_coords_idx[i]]
            self._masks.append(masking_strategy.generate(bd.Nc, self.num_patches, valid, coords,
                                                          subsampled=keep is not None))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, trial_idx):
        mask = self._masks[trial_idx]

        x, y = self.base_dataset[trial_idx]
        x_patches, time_indices = slice_patches(x, self.patch_len, self.patch_stride)

        coords_idx = self.base_dataset.trial_to_coords_idx[trial_idx]
        coords     = self.base_dataset.all_coords[coords_idx]
        valid_channels = self.base_dataset.all_valid_channels[coords_idx]
        keep = self._keep[trial_idx]
        if keep is not None:                       # subsampled window: removed channels become padding
            x_patches = x_patches * keep[:, None, None]
            coords = coords * keep[:, None]
            valid_channels = valid_channels & keep

        return x_patches, coords, mask, time_indices, y, valid_channels


class MontageBatchSampler(Sampler):
    """Batches of one montage (real-channel set) each, so a batch can drop the canonical
    channels none of its windows has (model/pretrain_loop.py's unpack_batch): most datasets fill
    only part of the 64-channel montage (Dreyer2023 27), and padded channels cost as much
    compute as real ones. Windows are shuffled within each montage, cut into batches, and
    the batch order is shuffled (shuffle=False: montages in order, windows in order)."""
    def __init__(self, dataset: PretrainDataset, batch_size: int, shuffle: bool):
        bd = dataset.base_dataset
        groups: Dict[tuple, List[int]] = {}
        for i in range(len(bd)):
            key = tuple(bd.all_valid_channels[bd.trial_to_coords_idx[i]].nonzero().flatten().tolist())
            groups.setdefault(key, []).append(i)
        self.groups, self.batch_size, self.shuffle = list(groups.values()), batch_size, shuffle

    def _batches(self):
        out = []
        for g in self.groups:
            g = [g[j] for j in torch.randperm(len(g)).tolist()] if self.shuffle else g
            out += [g[k:k + self.batch_size] for k in range(0, len(g), self.batch_size)]
        return [out[j] for j in torch.randperm(len(out)).tolist()] if self.shuffle else out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return sum(-(-len(g) // self.batch_size) for g in self.groups)


class FinetuneDataset(Dataset):
    """
    Wraps EEGDataset for supervised finetuning and trial-level inspection.
    Preserves original trial boundaries and labels.
    Yields: (x, coords, label, valid_channels, valid_length)
      x:              [C, T]
      coords:         [C, 3]
      label:          scalar
      valid_channels: [C] bool, True = real (not zero-padded) channel
      valid_length:   scalar int, real (non-padded) time length
    """
    def __init__(self, base_dataset: 'EEGDataset'):
        self.base_dataset = base_dataset
        n_classes = len(set(base_dataset.labels.tolist()))
        print(f"Initializing FinetuneDataset: {len(base_dataset)} trials, {n_classes} classes, shape {tuple(base_dataset.data.shape[1:])}.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, label = self.base_dataset[index]
        task_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[task_idx]
        valid_channels = self.base_dataset.all_valid_channels[task_idx]
        valid_length = self.base_dataset.all_valid_length[task_idx]
        return x, coords, label, valid_channels, valid_length
