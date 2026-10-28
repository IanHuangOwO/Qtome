# Plan (2026-10-28): dataset metadata generators -- one writer for the hand-written ones

Status: DRAFT, not started. Written 2026-10-28 after the code graph (graphify) flagged the community "Dataset Metadata Generators" as weakly
interconnected (cohesion 0.043) and `write_moabb_metadata` as a 43-edge hub. Nothing below is decided until the user answers section 6.

## 1. Why

- Every dataset has `datas/<split>/<Name>/gen_metadata.py`, the reproducible source of its `metadata.json`. Adding a dataset starts from
  `datas/_template_gen_metadata.py`, which is copied and edited.
- Two shared writers already exist in `IO/metadata.py`: `write_moabb_metadata` (MOABB-backed) and `write_bids_metadata` (BIDS-backed). Generators
  that use them are short. The rest still carry their own copy of the same schema code.

## 2. What is there (counted 2026-10-28, `git ls-files 'datas/*gen_metadata.py'`, archive excluded)

| Kind | Generators | Lines (all together) | Notes |
|---|---|---|---|
| MOABB-backed, call `write_moabb_metadata` | 38 | 987 (median 24 each) | already shared |
| BIDS-backed, call `write_bids_metadata` | 12 | 246 | already shared |
| hand-written in the template's shape (`ROOT`, `DATASET_INFO`, `CHANNELS`, `TARGETS`, `build_data_structure`, own `main()` with `json.dump`) | 27 | 2,871 (about 106 each; BETA_4s 195, ERP_Longitudinal 160) | the duplication |

Five archived generators under `datas/archive/` are out of scope. The 27 hand-written ones repeat the same `meta = {data_metadata: {dataset_name,
dataset_info, acquisition, targets, channels}, data_structure}` assembly and the `json.dump` to `metadata.json`. What differs between them is the part worth
keeping in each file: the `DATASET_INFO` text (source, reference, notes about quirks), the channel list, the targets, and `build_data_structure` (how the
raw files are laid out). Some channel lists are the same literal 64-channel 10-10 list copied from one dataset to the next.

## 3. Options

- **A. One declarative spec file per dataset** (JSON/TOML) read by one writer. Largest change: moves the `build_data_structure` logic and the long notes out of
  Python, and the file layouts differ too much to express as data.
- **B. A third shared writer, `write_files_metadata`, in `IO/metadata.py`** (recommended). The generator keeps its dataset-specific parts as data and one function
  (`build_data_structure`); the writer does the assembly, the required-field checks and the write. Same shape as the two existing writers, so no new idea.
- **C. Do nothing.** The duplication is stable and each file is independent; the cost is that a schema change touches 27 files.

## 4. Proposed work (option B)

1. Add `write_files_metadata(root, name, dataset_info, channels, targets, structure, acquisition, **extra)` to `IO/metadata.py`. It builds the same dict the
   hand-written `main()` builds and writes it with the same `json.dump(indent=4)`. It checks the REQUIRED fields of `docs/agents/adding-a-dataset.md` Step 3.
2. Put the repeated channel lists in one place (`configs/montages.json` already names sub-montages; a dataset whose list equals one names it instead of repeating it).
   Datasets with their own electrode positions keep their own `CHANNELS`.
3. Pilot on two datasets (one single-file layout, one multi-run folder layout). Check the result byte for byte (section 5) before touching the others.
4. Migrate the rest in batches of about five, each batch checked the same way.
5. Update `datas/_template_gen_metadata.py` and `docs/agents/adding-a-dataset.md` Step 3 to the new shape; remove the copy-paste main from the template.
6. Run `graphify update .` and compare the community cohesion with 0.043 as a rough sign that the structure improved.

Rough size, as an estimate and not measured: the assembly and write code is about 25 lines per file, so 27 files lose about 700 lines; the
dataset-specific text and `build_data_structure` stay.

## 5. How to check nothing changed

A migrated generator must write a `metadata.json` identical to the committed one (`diff` of the files, not just equal after parsing, because the key order
and `indent=4` formatting are part of the file). Regeneration needs the raw folder, so:
- datasets with `raw/` on disk: regenerate into a temporary path and `diff`;
- datasets without `raw/` on disk: feed the new writer the existing `metadata.json`'s own `data_structure` as `structure` and `diff` the output; the part the writer
  owns (assembly and formatting) is what changed, and `build_data_structure` is unchanged code that is moved, not edited.
Then run `python cache_dataset.py --config configs/compile.json --verify-only` for the migrated datasets.

## 6. Questions for the user

1. Option B (recommended), or something else?
2. Channel lists: name an existing montage in `configs/montages.json` when equal (recommended), or leave each file with its own literal list?
3. Keep the `DATASET_INFO` notes in the generator files (recommended: they record quirks found while adding each dataset)?
4. Pilot datasets: any preference, otherwise the shortest single-file one and BETA_4s (the longest)?

## 7. Not in scope

Loaders (`datas/*/loader.py`, `IO/loader.py`), the `metadata.json` schema, the five archived generators, and datasets whose raw data is not on disk beyond the check in section 5.
