# Plan (2026-10-24): analysis restructure -- one file per analysis, one cut-over

Status: DECIDED. Decided with the user 2026-10-24: (1) compute and draw live together in one file per panel;
(2) a new top-level `panels/` replaces `tools/analysis`, `tools/viz`, `tools/panels`; (3) presets are declared by each panel;
(4) one entrypoint `analyze.py`; (5) one big cut-over, a new start, no adapter period.

## 1. Why

- One analysis is spread over three folders (task filter: `model/Qlet/Qlet_filter_tools.py`, `tools/analysis/task_filter_eval.py`,
  two `tools/viz/task_filter_*.py`, five panels). A new analysis edits `analysis_pretrain.py` (flags, docstring), `tools/panels/__init__.py`
  (PRESETS), `CLAUDE.md`.
- Grab bags: `tools/viz/atom_plots.py` (506 lines) and `codebook.py` (464) hold unrelated figures; `tools/analysis/__init__.py` mixes
  config / `--set` / output paths (used by the training scripts) with analysis helpers (it caused the import cycle).
- Numbers-then-figure exists only for the task filter. Output paths are rebuilt in every panel.
- `BaseCodebookChecker` / `BasePlotter` (model hooks) are analysis-only since training draws nothing, yet `model/` imports `tools.viz`.

## 2. Target layout

```
analyze.py                      python analyze.py pretrain --run <bb> [--preset standard|quick] [--panel a b] [--redraw] [panel flags]
                                python analyze.py finetune --group base=<bb> ... ; python analyze.py list
panels/
  core/   registry.py (discovery by filename, presets collected from the panels), context.py (Ctx: config, paths, lazy model,
          device), numbers.py (save_numbers / load_numbers: json + npz, one convention), runner.py (compute -> draw, failures
          don't stop the rest, --redraw = draw only)
  pretrain/<name>.py            one analysis each (a folder <name>/ when it passes ~250 lines, e.g. task_filter/)
  finetune/<name>.py
  shared/<topic>.py             only code used by >= 2 panels (rule of two); moved here the day the second user appears
  kit/                          generic drawing primitives: topomap, timeseries, style (no analysis logic)
IO/config.py                    load_config, apply_overrides, deep_merge, model_output_root, resolve_output_path, QLET_OUTPUT
                                (what train_*.py and cache_feature.py import today from tools.analysis)
```

A panel file:

```python
NAME = 'atom_templates'; STAGES = {'pretrain'}; PRESETS = {'standard', 'quick'}
def add_args(p): ...                 # only this panel's flags (--share, --single, ...); `--panel X --help` shows just them
def compute(ctx): ...                # numbers -> ctx.numbers_dir (json / npz)
def draw(ctx): ...                   # reads those numbers -> ctx.fig_dir; one function per figure (draw_filter, draw_applied, ...)
```

Output folders unchanged: numbers `<bb>/pretrain/analysis/`, figures `<bb>/analysis/pretrain/`, finetune as today. Old results stay valid.

## 3. Cut-over procedure (verify before deleting anything)

1. **Baseline.** On `qlet_small_v4flat_onephase_s1` (and a finetune group of the onephase backbones) run every panel with small arguments
   (`--max-trials 40`, two datasets, task-filter cells that exist) and copy all outputs to a baseline dir. Run the old code twice:
   panels whose outputs differ run to run are marked numbers-only (tolerance) in the check.
2. **Build `panels/core` + `IO/config.py`**, repoint `train_pretrain.py`, `train_finetune.py`, `train_baseline.py`, `cache_feature.py`.
3. **Port the 27 panels** (each: one file, compute + draw, own flags, own preset tags). The logic is moved, not rewritten; layout and
   names are new. Model hooks: see decision A.
4. **Compare** every output byte-for-byte (json / npz / png) with the baseline; differences are fixed or explained one by one.
5. **Delete** `tools/analysis`, `tools/viz`, `tools/panels`, `analysis_pretrain.py`, `analysis_finetune.py`; update `CLAUDE.md` (commands,
   owner table), `README.md`, `docs/agents/adding-a-tool.md`, `adding-a-model.md`; add `docs/agents/adding-a-panel.md`; regenerate the
   graph. `tools/misc/` (pipeline utilities) stays; `tools/misc/profile_precision.py` and `summarize_runs.py` move or repoint.
6. One commit per step above, pushed after the comparison passes.

Risk: a one-shot cut-over has no fallback panel by panel, so the baseline and the byte comparison (step 1, 4) are the safety net.
Panels that need data absent here (some finetune cells) are compared on whatever cells exist and listed as unverified.

## 4. Decisions (all settled 2026-10-24)

- **A.** The model hooks `plotter_cls` / `codebook_checker_cls` are removed from `BasePlugin`; the `training_dashboard` and `codebook` panels own that
  code, and `model/` no longer imports `tools.viz`. A future backbone gets its own panel files; `adding-a-model.md` loses two steps.
- **B.** `panels/pretrain/` and `panels/finetune/`; a panel used by both stages shares through `shared/`.
- **C.** `panels/` and `analyze.py`.
- **D.** `--redraw` covers every panel; heavy panels (snapshots, codebook, atom maps) save the arrays they draw from as their numbers file.
