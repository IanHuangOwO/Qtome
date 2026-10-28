"""
Copy-paste starting point for a new dataset's metadata generator. Save as
datas/MyDataset/gen_metadata.py — run it to (re)produce metadata.json, the
reproducible source of truth for that dataset. See
docs/agents/adding-a-dataset.md Step 3 for the full schema and field-by-field
notes (REQUIRED fields, coordinate convention, raw/-prefixing, etc).
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))      # the repo root (datas/<split>/<Name>/ -> three up), for IO
from IO.metadata import write_files_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))

# TODO: fill in from Step 0 research (source_url, description, task_type,
# reference, contact, and any dataset-specific notes worth preserving).
DATASET_INFO = {
    "source_url": "...",
    "file_format": "...",   # e.g. "MATLAB (.mat)", "CSV", "EDF+", "GDF"
    "description": "...",
    "task_type": "...",
}

# TODO: one entry per channel, 1-indexed, SAME order as the raw signal's
# channel axis. "label" should be a standard 10-20/10-10 name where possible
# (see IO/dataset.py's _map_channels/_LABEL_ALIASES). "coordinates" is polar
# (angle in degrees, radius ~0-1) — omit per-channel if the label resolves via
# MNE's standard_1020 montage (see channel_xyz in IO/channels.py).
CHANNELS = {
    "1": {"label": "Fp1", "coordinates": {"polar_angle_deg": -17.926, "polar_radius": 0.51499}},
}

# TODO: one entry per class, 0-indexed, dense — must match the label ints
# your loader.py emits.
TARGETS = {
    "count": 2,
    "type": "...",   # free-form: motor_imagery / ssvep / p300_speller / ...
    "0": {"label": "..."},
    "1": {"label": "..."},
}


def build_data_structure(raw_dir):
    """Walks raw_dir and returns {subject_id_str: {...paths relative to
    datas/MyDataset/, always raw/-prefixed...}} — shape depends on how the
    raw files are laid out, see Step 4 in adding-a-dataset.md for the three
    common shapes (single file, split signal/label, multi-run folder)."""
    structure = {}
    for fname in sorted(os.listdir(raw_dir)):
        m = re.match(r"...", fname)  # TODO: match this dataset's filename pattern
        if not m:
            continue
        subject_num = int(m.group(1))
        structure[str(subject_num)] = {"file": f"raw/{fname}"}
    return structure


def main():
    structure = build_data_structure(os.path.join(ROOT, "raw"))
    # IO/metadata.py's write_files_metadata assembles data_metadata + data_structure, checks the REQUIRED fields and writes
    # metadata.json. Extra keys of your dataset go in after_info / after_targets / after_channels (where the file has them).
    write_files_metadata(
        ROOT, "MyDataset", DATASET_INFO,
        acquisition={
            "sample_frequency": 250,          # REQUIRED. Hz, used for filtering/resampling.
            "window_size_seconds": 4.0,       # REQUIRED if the loader needs to cut fixed windows.
            "num_subjects": None,             # None -> len(structure)
        },
        targets=TARGETS, channels=CHANNELS, structure=structure,
        channel_header={"system": "10-20 International System"})


if __name__ == "__main__":
    main()
