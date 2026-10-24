"""Alignment checks of the datasets (run once after construction, before training starts)."""
import torch
from typing import Dict

from IO.wrappers import FinetuneDataset, PretrainDataset


def sanity_check_base(base_dataset: 'EEGDataset') -> None:
    """
    Verifies EEGDataset's parallel per-trial/per-task arrays stayed aligned
    through loading/padding/windowing. Call once after construction, before
    training starts — cheap (no data copies), catches an alignment bug (wrong
    label/coords/subject_id paired with a trial) that would otherwise silently
    corrupt training instead of crashing anywhere obvious.
    """
    n = len(base_dataset)
    if n == 0:
        raise ValueError("EEGDataset has 0 trials.")

    for name, arr in (('labels', base_dataset.labels), ('subject_data', base_dataset.subject_data),
                       ('dataset_names', base_dataset.dataset_names),
                       ('trial_to_coords_idx', base_dataset.trial_to_coords_idx)):
        if len(arr) != n:
            raise ValueError(f"EEGDataset.{name} has length {len(arr)}, expected {n} (== len(data)).")

    n_tasks = len(base_dataset.all_coords)
    for name, arr in (('all_valid_channels', base_dataset.all_valid_channels),
                       ('all_valid_length', base_dataset.all_valid_length)):
        if len(arr) != n_tasks:
            raise ValueError(f"EEGDataset.{name} has length {len(arr)}, expected {n_tasks} (== len(all_coords)).")

    max_idx = max(base_dataset.trial_to_coords_idx)
    if max_idx >= n_tasks:
        raise ValueError(f"trial_to_coords_idx references task {max_idx}, but only {n_tasks} tasks were loaded.")

    for coords in base_dataset.all_coords:
        if coords.shape != (base_dataset.Nc, 3):
            raise ValueError(f"A coords entry has shape {tuple(coords.shape)}, expected ({base_dataset.Nc}, 3).")
    for vc in base_dataset.all_valid_channels:
        if vc.shape != (base_dataset.Nc,):
            raise ValueError(f"A valid_channels entry has shape {tuple(vc.shape)}, expected ({base_dataset.Nc},).")

    # dataset_names and trial_to_coords_idx are built in the same per-task loop
    # (EEGDataset.__init__), so every trial mapped to task index T must report the
    # same dataset_name every time it recurs -- a scramble (wrong zip/append order)
    # breaks this even though the plain length checks above still pass.
    task_dataset_name: Dict[int, str] = {}
    for i in range(n):
        task_idx = base_dataset.trial_to_coords_idx[i]
        name = base_dataset.dataset_names[i]
        seen = task_dataset_name.setdefault(task_idx, name)
        if seen != name:
            raise ValueError(
                f"Trial {i}: dataset_name {name!r} disagrees with an earlier trial mapped "
                f"to the same task index {task_idx} ({seen!r}) — trial_to_coords_idx and "
                f"dataset_names are out of sync."
            )

    if not torch.isfinite(base_dataset.data).all():
        raise ValueError("EEGDataset.data contains NaN/Inf — check upstream cache/normalize.")

    print(f"[sanity check] EEGDataset: {n} trials, {n_tasks} tasks — "
          f"data/labels/coords/subject/dataset_name alignment OK.")


def sanity_check_wrapper(dataset) -> None:
    """
    Pulls the first and last item through __getitem__ on the actual training
    dataset (PretrainDataset/FinetuneDataset) — catches an
    index-mapping bug in the wrapper itself (e.g. a masking-strategy resolve()
    or trial_to_coords_idx lookup gone wrong) before training starts, rather
    than a cryptic mid-epoch crash or, worse, silently wrong data with no
    crash at all.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError(f"{type(dataset).__name__} has 0 items.")
    for idx in sorted({0, n - 1}):
        item = dataset[idx]
        for t in item:
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                raise ValueError(f"{type(dataset).__name__}[{idx}] contains NaN/Inf.")
    print(f"[sanity check] {type(dataset).__name__}: {n} items, first/last __getitem__ OK.")
