# CLAUDE.md

Guidance for Claude Code in this repository. Canonical terms: `CONTEXT.md`. Design history: `docs/adr/` (decisions),
`docs/cards/` (pre-registered experiments), `docs/reports/` (results) -- kept locally, not tracked in git.

**Name:** the model is **Qlet** since 2026-10-27 (it was Qtome, MeSAE before 2026-10-06); its unit is the **qlet** (it was Q-atom; `atom` in code identifiers). Code, folders (`model/Qlet/`, `configs/Qlet/`, `output/Qlet/`), `model_params.Qlet` and run names (`qlet_*`) were renamed the same day. Checkpoints and saved configs from before still carry `model_type` `Qtome` / `model_params.Qtome`: `IO/config.py` and `model/factory.py` read both. The phoneme-like framing is dropped (qlets act as a filterbank). The conda env, the repo folder and the GitHub repo keep the name `qtome`.

## Environment

Run everything with the `qtome` conda env: `/home/mamechin/anaconda3/envs/qtome/bin/python` (torch, mne,
moabb). A bare `python` is `base`, which has no `mne`/`moabb`: coordinates silently fall back to flat polar
values from `metadata.json` and MOABB loaders fail. The machine is CPU-bound (20 cores, one 12 GB GPU):
cap threads (`--threads`, `training_params.<mode>.num_threads`). Parallel jobs: 3 for unattended (overnight) queues,
2 while the user is working. Measured 2026-10-02 on tiny pretrains: 3 jobs give ~10% more total throughput than 2
(tokenizer epoch 127 s vs 93 s per job) but each job runs ~35-40% slower; three pretrains use ~8 of 12 GB GPU memory.

## Commands

```bash
# Compile raw datasets into per-subject .npz caches (bandpass + resample baked in; verification runs after;
# --dataset NAME narrows it, --verify-only / --deep for checks). See docs/agents/adding-a-dataset.md.
python cache_dataset.py --config configs/compile.json

# Pretrain: one run, masked phase only since 2026-10-21 (tokenizer_epochs 0; the unmasked tokenizer phase of docs/adr/0013
# is still available with tokenizer_epochs > 0)
python train_pretrain.py --config configs/runs/<backbone>/pretrain.json

# Finetune a head on the frozen backbone (atom-code cache built on first use by cache_feature.py)
python train_finetune.py --config configs/runs/<backbone>/finetune/<head>/<cell>.json

# From-scratch baseline (model/EEGNet/, settings configs/EEGNet/settings.json, Compass-checked) on one cell, same
# splits and group_eval.json as a Qlet cell -> output/EEGNet/finetune/compass/<dataset>_<mode>_seed<k>/
python train_baseline.py --model EEGNet --dataset BNCI2014004 --protocol mi_fewshot

# Any config value can be overridden (dotted path, JSON value, repeatable; artifacts/config.json records it)
python train_finetune.py --config <cfg> --set training_params.finetune.learning_rate=0.003

# Experiments: a sweep file (base config + cases x grid, optional "backbones") -> queue plan; run_queue runs it
# resumably (state in output/queue/<plan>/: .done/.failed/.log per job; plan lines 'job <name> :: <cmd>' / 'wait')
python -m tools.misc.sweep configs/Qlet/sweeps/<sweep>.json > output/queue/<plan>.plan
python -m tools.misc.run_queue output/queue/<plan>.plan --max-parallel 2 --threads 8

# Analysis: one file per analysis under panels/<stage>/ (compute the numbers, then draw the figures from them; failures don't stop the
# other panels; presets are tagged in each panel's PRESETS). Pretrain = one backbone -> numbers in
# output/Qlet/<backbone>/pretrain/analysis/, figures in output/Qlet/<backbone>/analysis/pretrain/ (standard: backbone_eval,
# attention_range, atom_usage, ridge_probe, atom_vs_raw, atom_templates, atom_duplicates, atom_labels, snapshot,
# training_dashboard, task_filter_artifacts (skipped without the three loso impute64 cells; TUAR eye / muscle / electrode / line-noise test of the multi-task filter); quick: attention_range, atom_usage, ridge_probe, atom_vs_raw, atom_templates, atom_duplicates; on request: atom_distribution, codebook (each loads every pretrain dataset: ~20-30 min),
# coord_robustness, long_context, profile, other task_filter_*; finetune: atom_maps,
# probe_maps, class_snapshots, seed_equivalence, training_dashboard). Judge each change by its own mechanism metric, not finetuning
# (ADR 0020). `--redraw` draws again from the saved numbers; `--panel <name> --help` shows a panel's own options. Writing a panel:
# docs/agents/adding-a-panel.md.
python analyze.py list                                                  # every panel, its presets and one-line description
python analyze.py check                                                 # every panel imports, flags consistent (no data; run after editing a panel)
python analyze.py pretrain --run <backbone> [--preset quick] [--panel <name> ...] [--redraw]
python analyze.py pretrain --panel profile [--train]                    # parameter counts + timing, no checkpoint
# Group = several backbones compared in one folder -> output/Qlet/<common name of the backbones>_analysis/ (--out overrides). A group name is <recipe>_s<seed> (s1, small_s2) or a bare s1;
# the recipe is the name without the seed, so one run can hold a recipe's seeds (qlet_tiny_v4flat_onephase_analysis) or several recipes (mean +- SE over each recipe's seeds).
# The stage was called `finetune` until 2026-10-27 (that name still works). Standard panels: summary, atom_seed_match (all seed pairs; skipped with one group), and from the numbers the pretrain panels saved under each backbone: backbone_eval (masked MSE per mask), atom_labels
# (ICLabel classes per seed and their agreement across seeds), noise_removal (TUAR / TUEV artifacts and brain kept), noise_removal_synthetic (artificial noise at TUAR strength), noise_classify (accuracy after
# removal); on request: report, gradient_correlation, subspace_affinity, pretrain_alignment, asr_cutoff (both ASR variants against their cutoff; needs task_filter_artifacts run with --art-asr-cutoffs 3 5 10 --art-atom-cutoffs 1 2 3 5). The noise figures show the general methods and the one atom removal that matches each noise (the multi-task filter and the other atom groups stay in the .md tables); --with-multitask also draws the earlier all-methods version (<name>_with_multitask.png). TUAR's electrode column scores the annotated bipolar pair (task_filter_artifacts reads the pair from the annotation; TUEV keeps the pooled peak). The noise panels need the pretrain panels backbone_eval, atom_labels, task_filter_artifacts (--art-dataset TUAR and TUEV), task_filter_noise_removal_synthetic and task_filter_atom_classify run on every
# backbone first (the last three are on request, about 2 h per backbone together). Figures that describe one backbone stay in <backbone>/analysis/; figures that compare backbones or seeds live here. The three noise panels on a backbone (task_filter_artifacts, task_filter_noise_removal_synthetic, task_filter_atom_classify) save numbers only; `--redraw` draws a single backbone's figure from them.
python analyze.py group --group base=<backbone> --group X=<backbone> --ref base [--head frozen_learned] [--panel <name> ...]
# Cross-task gradient panels (on request, per backbone; EEG-FM-Bench Figs. 3, 4, 8): backbone gradients of each finetune cell's trained head, shared probe cached as gradient_probe_<group>.npz
python analyze.py group --group base=<backbone> --head <head> --panel gradient_correlation --panel subspace_affinity --panel pretrain_alignment [--cells <cell> ...]
python -m panels.group.summary 'output/Qlet/<backbone>/finetune/<head>/*' [--ref <backbone>:<head>] [--rank N]   # standalone table

# Task filter: a reconstruction keeping only what classification heads use (panels/shared/task_filter/fit.py). Panels of
# analyze.py pretrain, run on request (not in a preset). --filter-source 'multi' (multi-task filter: MI + P300 + SSVEP
# heads) or one dataset (single-task filter); needs loso cells <ds>_impute64_loso_seed1 (impute_missing_channels=true).
# Numbers: output/Qlet/<backbone>/pretrain/analysis/filter/<source>/ (filter.pt, eval_*) and .../compare/; figures
# (filter_analysis_*, filter_<ds>_*, compare_*.png): output/Qlet/<backbone>/analysis/filter/<source>/ and .../compare/
python analyze.py pretrain --run <backbone> --panel task_filter_build               # fit and save the filter (again)
python analyze.py pretrain --run <backbone> --panel task_filter_filter [--share 0.1]   # the filter alone: importance maps, power kept of white input
python analyze.py pretrain --run <backbone> --panel task_filter_eval [--filter-test mi p300 noise] [--target <ds> ...]
python analyze.py pretrain --run <backbone> --panel task_filter_panel --target <ds> [--share 0.1]
python analyze.py pretrain --run <backbone> --panel task_filter_artifacts [--art-shares 0.5 0.1]   # the filter on TUAR's annotated eye movement / muscle / electrode artifacts and its real 60 Hz line noise, each with a clean-window control, against ASR / ICLabel, nothing added
python analyze.py pretrain --run <backbone> --panel task_filter_compare --target <ds> [--single <source ds>]   # multi-task filter vs ASR / ICLabel (single-task filter only with --single: it lost)
```

No test suite: modules carry runnable self-checks (e.g. `head.feature_head._selfcheck_head_modules()`), and
validation runs during training. Refactors here are verified by reproducing recorded results bit-identically.

### Long-running jobs: always monitor exit and errors

Past failures: a watcher waiting on a log line that never came sat silent for 7 h and blocked the step
chained behind it; a finished queue went unreported for 40 min.
- Launch so the exit is noticed: an agent uses the tool's background mode (exit notification), not a detached
  `nohup ... &`. If detaching is unavoidable, add a background watcher on the PID.
- Wait on a process exiting (`while kill -0 $PID`) or a file existing, never on a specific log line.
- Every job records its exit code; a chained step checks the previous step's status.
- Report a job's end right away (success, or failure + last error lines). On a status check, list anything
  that died or went quiet.
- Once a queue has finished and its outcome is reported, delete its state (`output/queue/<plan>/`, `.plan`,
  `.log`): results live in each run's own output dir. Keep it only while a job failed and needs a rerun.

## Configs

Configs are grouped by owner: shared ones at the top of `configs/`, model-specific ones in `configs/<Model>/`.
`configs/Qlet/pretrain.template.json` / `configs/Qlet/finetune.template.json` are starting points, never
run directly: copy to `configs/runs/<backbone>/pretrain.json` and `configs/runs/<backbone>/finetune/<head>/<cell>.json`
(a run config is one complete JSON file: copy a template, edit it).
`configs/runs/` is gitignored; every run's effective config is saved in its `artifacts/config.json`.
The corpus list is the same for every size (generated by `tools/misc/build_pretrain_corpus.py`: 46 datasets, 6,089 subjects, 210 h; group share = mean of the target and the
sqrt(subjects) share, cohort hours ~ unique subjects^0.75, equal time per subject inside a dataset through
`dataset_params.pretrain.<ds>.max_windows_per_subject`, a dataset over 1,000 subjects keeps a seeded pick of 1,000); the size is one
setting, `preprocess_params.window_fraction`: tiny 0.05 / small 0.2 / medium 0.5 / large 1.0 (the template's value)
of every subject's windows, nested at one `window_fraction_seed`. Set it in the run config or with `--set`.
Shared: `configs/compile.json` (compile params + dataset list), `configs/montages.json` (canonical and
sub-montages), `configs/finetune_protocols.json` (evaluation splits per protocol, every model). Qlet:
`configs/Qlet/` (templates, `protocol_heads.json` = its head and optimiser per protocol, `sweeps/`). EEGNet:
`configs/EEGNet/settings.json`. See `configs/README.md`.

Key fields (templates show defaults):
- `model_params.Qlet.pretrain`: `patch_len`, `embed_dim`, `enc_depth`, (the encoder is `enc_depth` blocks at full patch
  resolution; pooling of the patch axis, its upsampling and the gated skips -- `blocks_per_stage` -- were removed
  2026-10-13: the user adopted the flat encoder on one seed, masked MSE -4.3%, docs/cards/2026-10-12-no-pooling.md; the
  pooled backbones, up to `qlet_tiny_v4tr2`, can no longer be rebuilt, their results stay in their output dirs),
  (the skip drop-path `skip_drop` was removed 2026-10-12: without it masked MSE fell 12% and no downstream cell lost
  over 3 seeds, docs/cards/2026-10-11-skip-drop.md; a `skip_drop` key in an old build_config is ignored),
  (time position: FoPE on every block's temporal attention, the only option since 2026-10-07 -- it replaced the learned
  absolute time table and the `temporal_bias` lag bias; checkpoints from before, v2 / v3, can no longer be rebuilt,
  their results stay in their output dirs; `time_idx` is still passed but unused by Qlet; block order is sequential,
  temporal -> spatial -> FFN: parallel attention was screened and rejected 2026-10-07),
  `spatial_heads`, `moe_ffn`, `atom_bank` (`n_atoms`, `hidden_width`), `loss` (the nested reconstruction loss, the only one since 2026-10-17:
  per patch, the error of the strongest 3 / 6 / 12 / 24 qlets weighted 1/8, 1/8, 1/4, 1/2 -- `nested_sizes` /
  `nested_weights`; it replaced the MP loss, removed with `mp_weight`, docs/adr/0018; spectral whitening of the error
  was tested and removed 2026-10-18), `spatial_embedding` (true: Fourier electrode-coordinate
  embedding + per-block directional relative spatial bias; false: neither -- the spatial ablation).
- `preprocess_params`: `canonical_channels` (a `montages.json` name or a list; every EEG channel
  of the dataset is kept, non-10-10 ones in free slots, > 64 channels reduced to the 64 sites, coordinates = MNE's
  template position for the channel name -- docs/adr/0023; the 10-10-only `grid` layout and the datasets' own
  `recorded` positions were removed 2026-10-07), `window_length`,
  `window_min_real`, `window_fraction`, `patch_length` 50, `patch_stride` 25 (50% overlap), `sample_freq` 200,
  `bandpass_filter` (input normalisation: per-trial z-score, the only option), and `mask` (`IO/masking.py`): `masking_strategy` `mixture` (one
  MaskMode per window -- channel_cluster / random_channel / time_block / random_token -- on a shared ratio
  ramp; the default) or `random` (the masking baseline); masks redrawn every masked epoch; `time_run` 2: `random_token`
  masks runs of 2 patches per channel (a lone patch leaks through the overlap; 2 hide 0.25 s fully; 3 before 2026-10-13); `time_block` `max_blocks` 2 (templates since 2026-10-05; the
  v2 backbones predate it): 1-2 holes splitting the ratio at random cut points, each >= 3 patches; `subsample` removes channels down to a
  `montages.json` sub-montage for part of the dense-cap windows (`prob` 0.1 in the templates since 2026-10-06, the
  user's choice without a test; earlier backbones used 0.2). Draw probabilities of the four masking methods (adopted
  2026-10-13 on 3 flat seeds, docs/cards/2026-10-12-token-runs-plus-time-block.md: token-run MSE -18%, channel masks -2%,
  time holes +1 to +4%, loso level): channel cluster 0.35, random channels 0.35, token runs (`random_token`) 0.20, time
  blocks 0.10; maximum ratios 0.3 / 0.3 / 0.4 / 0.4. Before: time blocks 0.30 and no token runs (cluster ratio 0.2
  before 2026-10-06). Channel modes take `exact` (templates since 2026-10-15): the count is rounded down, and the
  cluster stops at it (it used to overshoot, up to 68% of BCI-22 at ratio 0.3) and applies only when it can hide >= 2
  channels (floor(ratio x C) >= 2). Mode
  `montage` (MontageMask, not in the template, never evaluated, card closed 2026-10-18): channels outside a sparse
  montage hidden as targets, replacing the padding subsampling (docs/cards/2026-10-15-sparse-montage-mask.md).
- `training_params.pretrain`: `model_name`, `output_path`, `epochs`, `tokenizer_epochs` (0 = one phase, adopted 2026-10-21 on 3 small seeds: masked MSE
  level, finetune level, 30-45 Hz coding worse, docs/cards/2026-10-19-one-phase.md; the mask ramp is 20 epochs, subsampling from epoch 21), `freeze_atoms`,
  `warmup_epochs`, `batch_size`, LR fields, `train_val_split`, `seed`.
- `model_params.Qlet.finetune` (the head, validated at build): `features` = a list of `{"type": <entry>,
  <per-entry keys>}`; entries are classes in `model/Qlet/head/entries.py`'s `ENTRY_TYPES` (`atom_power`, `signed_ab`,
  `latent_signed`, and the raw-input baseline `raw_band` / `raw_signal`; `atom_band` / `evoked` / `phase_advance` /
  `latent_power` removed 2026-10-07) -- a new
  head feature is one class + one registry line. Every entry has its own spatial filter. Per-entry keys
  (`spatial_k`, `time_pool`, `time_rank`, `window`, `atom_rank`, `latent_proj`) may also be set
  at the top level as defaults; plus `dropout`. Defaults: `_HEAD_DEFAULTS`.
- `training_params.finetune`: `pretrained_checkpoint`, `protocol` (a `configs/finetune_protocols.json` split + its `configs/Qlet/protocol_heads.json` head:
  mi_loso / mi_fewshot / p300_loso / p300_fewshot, tuned on DEV sets BNCI2015001 / BNCI2014009 only; each sets
  the combined head; applied over the merged config, so a head in the run config is overridden -- `--set` still wins), `split` = `{"type": ...}` from `train_finetune.py`'s `SPLITS`:
  `loso`, `subject_kfold` (`n_folds`), `eval_subjects`, `kfold`, `blocked_kfold`, `fewshot` (`train_fraction`,
  EEG-FM-Compass calibration); all take `sessions` and `seed`, per-subject types also `purge` (P300 overlap);
  unknown keys are rejected. Plus LR fields, `epochs`, `class_weight` (`balanced`), `batch_size`, `seed`.
  `fit`: `sgd` (default) or `closed_form` (`model/Qlet/head/closed_form.py`, no SGD). Few-shot always uses the closed-form
  structured head: the all-atom head's own factors (spatial filter x qlet weights x time course) set in closed
  form, one head for every paradigm, signed half at full time resolution (the protocols set it;
  docs/reports/2026-10-06-structured-fewshot-head.md); since 2026-10-15 both halves use the Fisher score and full time
  rank (`power_score` / `signed_score` `fisher`, `time_rank` / `signed_time_rank` `full`; DEV +4.6, Nakanishi2015
  few-shot 86.3 -> 96.7; docs/reports/2026-10-15-closed-form-unified.md). (The per-paradigm CSP / xDAWN / TRCA heads it replaced were removed 2026-10-07.)
  Loso uses SGD (closed-form loses there). SSVEP DEV sets: Kalunga2016 (not phase-locked), Wang2016_dev (phase-locked,
  in the pretraining corpus as unlabelled windows).

## Who owns what

| Owner | Code | Configs |
|---|---|---|
| shared (every model) | `cache_dataset.py`, `datas/`, `IO/` (incl. `IO/splits.py`: protocol splits, `make_runs`), `train_pretrain.py` (dispatches through `model/factory.py`'s plugin registry), `model/base_*.py`, `model/factory.py`, `IO/config.py`, `tools/misc/` | `compile.json`, `montages.json`, `finetune_protocols.json` |
| Qlet | `model/Qlet/` (incl. `finetune/`: head sources, protocol, fit, atom cache), `train_finetune.py` + `cache_feature.py` (entry points), `analyze.py` + `panels/` (every analysis and figure, incl. the task filter) | `configs/Qlet/` |
| from-scratch baselines | `model/EEGNet/`, `train_baseline.py` | `configs/EEGNet/` |

A new pretrained backbone follows `docs/agents/adding-a-model.md` (plugin); a new from-scratch baseline is
`model/<Name>/<Name>.py` + a line in `train_baseline.py`'s `BASELINES` + `configs/<Name>/settings.json`.

## Architecture

**Qlet**: an EEG tokenizer. A TSA encoder feeds a static qlet dictionary (AtomBank: every qlet active at
every patch; each reconstructs a patch as `a*D + b*H`, D a template and H its quadrature partner), trained by
masked reconstruction (nested loss: qlets ranked per patch by strength, docs/adr/0018). Routed (top-k) qlets were removed (docs/adr/0022; code on the `routed-stamps`
branch). The default downstream head reads only the qlets: qlet power (`atom_power`, induced band power: MI) and the
signed qlet gains (`signed_ab`: phase-locked / P300). The stronger head swaps `signed_ab` for the signed encoder
output z (`latent_signed`, PCA): level on loso, better on MI few-shot (BNCI2014004 +3.5)
(docs/reports/2026-10-02-head-ablation.md, docs/reports/2026-09-29-combined-head.md). Plugged in via `model/Qlet/plugin.py` (`model/factory.py`
`MODEL_REGISTRY`, docs/adr/0004).

```
datas/<split>/<Name>/loader.py  compile time only; MOABB datasets have none (metadata's moabb block -> IO/loader.py's MoabbLoader)
 └ cache_dataset.py    bandpass + resample each CONTINUOUS recording (BaseSubjectLoader._filter_run),
 │                     then cut epochs (event-anchored [event-1 s, event+4 s) for pretraining; the reported finetune
 │                     sets use EEG-FM-Compass post-event windows via metadata moabb.onset_window/window, or fixed windows); drops
 │                     flat-line dropout windows -> datas/<split>/<Name>/cache/*.npz. BETA_3s/4s ship
 │                     pre-epoched and are filtered per epoch.
 └ IO/dataset.py       EEGDataset maps channels onto the 64 slots (non-canonical channels in free slots; missing -> zero
 │                     padding, valid_channels marks real ones), normalises per trial; PretrainDataset cuts
 │                     windows -> patches and draws masks (IO/wrappers.py); MontageBatchSampler makes one-montage batches
 └ train_pretrain.py   masked phase from epoch 1 (tokenizer_epochs > 0 first adds the old unmasked, temporal-only phase)
                       (spatial attention + coordinate embedding on, mask curriculum starts)
```

- Backbone, one file each in `model/Qlet/`: `embeddings.py` (`SpatialTemporalEmbeddings`, `RelativeSpatialBias`, FoPE), `ffn.py` (MoE FFN),
  `encoder.py` (`TSABlock`: temporal attention -> spatial attention -> MoE FFN, LayerScale; `TSAEncoder`: 8 blocks at full patch
  resolution, no pooling; padded channels and padded tail patches are masked out of attention), `atom_bank.py` (`AtomBank`),
  `patches.py` (overlap-add, fold). Finetune side in `model/Qlet/head/`: `extractor.py` (`AtomExtractor`: qlet codes and optionally z
  from the frozen backbone), `entries.py` (the `ENTRY_TYPES` registry), `config.py` (defaults, validation, widths), `pooling.py`,
  `feature_head.py` (`FeatureHead`).
- `Qlet.py`: `QletPretrain` (phases, per-sample masked loss: a sample counts as masked only if every patch
  covering it is masked), `FinetuneModel`.
- `model/EEGNet/EEGNet.py`: the EEGNet baseline (Compass's architecture), trained from scratch by `train_baseline.py`;
  not a plugin (no backbone). Finetune caches store volts: raw-input baselines multiply by 1e6.
- `factory.py`: a pretrain checkpoint stores its `build_config`; `build_from_checkpoint` rebuilds a trained
  backbone from it, never from the editable run config.
- `train_pretrain.py`: subject split by `IO/splits.py`'s `split_pretrain_subjects` -- person-disjoint per
  cohort (`metadata.json` `data_metadata.cohort`) and independent of dataset order; a subject's near-flat
  channels (std < 0.10 x median, dead electrodes / the recording reference) are treated as padding.

**Sparsity budget, a hard ceiling:** each active qlet gives two free scalars per channel, so keep
`2 * n_atoms < patch_len` with margin (template: 24 qlets since 2026-10-09, 48 < 100; 16 / 24 / 32 compared over 3
seeds in docs/cards/2026-10-08-atoms-32.md: 24 keeps most of 32's hidden-channel and high-frequency gain at half its
SSVEP cost). Past it, the active slots fit any patch
regardless of the templates and it stops being sparse coding (measured at DOF 56 > 50: recon MSE ~0 on every
dataset, kurtosis 6.7 -> 1.2). docs/adr/0011.

## Outputs

Each model has its own output folder: a run writes to `output/<model_type>/<training_params.<mode>.output_path>`
(default `<model_name>/pretrain` for pretrain; `model_type` defaults to Qlet; `IO/config.py` QLET_OUTPUT). EEGNet:
`output/EEGNet/finetune/compass/<dataset>_<mode>_seed<k>/`.
Per Qlet backbone: `output/Qlet/<backbone>/pretrain/` (`checkpoint/last.pth` -- prefer it over `best.pth`, which locks
onto an easy epoch of the mask curriculum; `artifacts/config.json`, `artifacts/history.json` (per-epoch curves + `epochs`: lr, seconds, phase, mask curriculum),
`artifacts/run_info.json` (code version, host, GPU, sizes, status), `artifacts/epoch_snapshots.npz` (atom templates per epoch),
`checkpoint/epoch_<n>.pth` every `training_params.pretrain.checkpoint_every` epochs (default 10);
`analysis/` numbers and tables; `feature_cache/` -- regenerable, keyed by checkpoint, build_config, code hash and data
fingerprint) and `output/Qlet/<backbone>/finetune/<head>/<cell>/` (`artifacts/group_eval.json`: per-subject `tail` = mean
of the last 10 epochs' balanced accuracy, `last`, kappa; `finetune/run_<fold>/` = `head.pth`, `history.json` (curves, lr, per-subject bal. acc. and kappa every epoch), `predictions.npz`
(eval trial indices, labels, last-epoch logits)). Runs store
logs, numbers and checkpoints only: **every figure lives in `output/Qlet/<backbone>/analysis/`** (`pretrain/` = the
pretrain panels, incl. `training_dashboard.png`, `snapshot/`; `filter/` = the task-filter panels; `finetune/<head>/<cell>/<fold>_training_dashboard.png`),
drawn afterwards from the stored numbers by the analysis panels (`--panel training_dashboard`, `--panel snapshot`);
training draws nothing. `output/Qlet/<name>_analysis/` (name = what the backbones' names share, e.g. `qlet_small_v4flat_onephase_analysis`) holds generated multi-backbone analysis (regenerable),
`output/queue/` queue state, `output/Qlet/archive/<date>_<topic>/` superseded experiments. Once a result is final, write it up
in `docs/reports/` (fixed format and index in `docs/reports/README.md`; copy the figures, tables and
pipeline scripts it cites) and delete the generated `*_analysis/` and queue folders.

## Data

`datas/<split>/<name>/metadata.json`: `data_metadata` (`acquisition.sample_frequency`, 1-indexed `channels`
with labels and, where the dataset ships them, its own positions -- polar `coordinates` with
`polar_equator_radius`, or `xyz`; per-subject `channel_xyz` in `data_structure` -- not read by training (template positions only);
`event_onset_seconds`, `moabb` class/kwargs/window, optional `cohort`) and
`data_structure` (per-subject files or `moabb_subject`). `datas/DATASETS.md` lists every dataset; regenerate
with `python -m tools.misc.dataset_inventory`. Read `docs/finetune-caveats.md` before reporting finetune numbers.

## Agent docs

- Issues: GitHub Issues (IanHuangOwO/Qtome) via `gh`; labels `needs-triage` / `needs-info` /
  `ready-for-agent` / `ready-for-human` / `wontfix` -- `docs/agents/issue-tracker.md`, `triage-labels.md`.
- Adding a dataset / model / montage / panel: `docs/agents/adding-a-*.md`. Raw downloads (NEMAR / MOABB, resumable,
  checksummed): `python -m tools.misc.fetch_datasets`. `panels/` = every analysis and
  figure (one file each, run by `analyze.py`, `docs/agents/adding-a-panel.md`); `tools/misc/` = pipeline utilities run directly.
- `.reshape(`/`.view(` silently scrambles data when it merges non-adjacent axes (it has hit training data
  three times): check any new one against `docs/agents/reshape-pitfalls.md`.
