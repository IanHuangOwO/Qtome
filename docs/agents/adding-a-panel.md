# Adding a panel (an analysis and its figures)

One analysis = one file under `panels/<stage>/`, run by `analyze.py`. It computes its numbers, saves them, then draws its figures from
the saved numbers, so `--redraw` never recomputes. (This replaced `tools/analysis` + `tools/viz` + `tools/panels`, 2026-10-24,
`docs/plans/2026-10-24-analysis-restructure.md`.)

```
analyze.py                 python analyze.py list | pretrain ... | group ...   (`finetune` = the old name of group)
panels/core/               registry (discovery by filename), context (Ctx), flags, numbers (save_json / save_npz), runner
panels/pretrain/<name>.py  one analysis of one backbone (a folder <name>/ with __init__.py when it outgrows ~250 lines)
panels/group/<name>.py     one analysis across backbones (--group NAME=BACKBONE ...: seeds / recipes): their finetune runs or the numbers the pretrain panels saved (panels/shared/groups.py)
panels/shared/<topic>.py   code used by two or more panels (rule of two; until then it stays in its panel); a topic that grows
                           several files becomes a folder (shared/task_filter/: cells, backbone, fit, context, tests)
panels/kit/                generic drawing primitives (topomap, timeseries, channel order); no analysis logic
IO/config.py              config loading, --set overrides, run output paths (the training scripts use these too)
tools/misc/                pipeline utilities run directly (sweep, run_queue, corpus and inventory builders), not analysis
```

## The panel contract

```python
"""name: one line saying what it computes and where the files go (this line is shown by `analyze.py list`).
...longer description, options..."""
PRESETS = {'standard', 'quick'}      # which presets run it; empty or absent = on request only
NEEDS_CHECKPOINT = True              # pretrain only; False when it runs without a backbone (profile)

def add_args(flags):                 # optional: this panel's own options
    flags.add('--share', type=float, default=0.1, help='...')

def compute(ctx): ...                # optional: numbers -> ctx.numbers('name.json' / 'name.npz'), via panels.core.numbers
def draw(ctx): ...                   # optional: reads those numbers, writes figures to ctx.figure('name.png')
```

- `Ctx` (panels/core/context.py): `stage`, `args` (the common options plus every selected panel's own), `device`, `cmap`, `numbers(*parts)` /
  `figure(*parts)` (a path under the numbers / figure folder, its directory created). Pretrain: `config`, `checkpoint`, `model` (built from
  the checkpoint's `build_config` on first use). Finetune: `groups` ({name: backbone}), `ref`, `head`.
- Folders: pretrain numbers `output/Qtome/<bb>/pretrain/analysis/`, figures `output/Qtome/<bb>/analysis/pretrain/`; finetune: both `--out`.
- Two panels may declare the same option (`--max-trials`, `--datasets`) if they agree on the default. `--panel X --help` shows only X's options.
- A panel that is only numbers (a table, a JSON) has no `draw`; one that only draws stored history (`training_dashboard`) has no `compute`.
- `--redraw` skips `compute`: a heavy panel saves the arrays its figures read (snapshots save the trial and the Q-atom readout, the codebook its
  usage arrays, ...), never a live model.
- A failing panel prints its traceback and the rest still run; the exit code is 1 if any failed.

## Where analysis code lives

Code that only analysis uses belongs under `panels/`, never in `model/`. A panel may import from `model/` and `IO/`; `model/`,
`IO/` and the training scripts never import from `panels/`. (The task-filter tools sat in `model/Qtome/` until 2026-10-24 for lack of
this rule; they are now `panels/shared/task_filter/`.)

## Procedure

1. Copy the closest panel in `panels/<stage>/` (a numbers-only one: `atom_usage`; numbers + figure: `atom_templates`; many figures: `codebook/`).
2. Write `compute` and `draw`; keep drawing functions pure (`plot_x(out_path, arrays...)`: no model, no dataset).
3. Needs something a sibling panel has? Import it only if that is already in `panels/shared/`; if it sits inside the sibling, move it to
   `panels/shared/<topic>.py` in the same change (that is the second user), and keep the sibling working.
4. `python analyze.py check` (no data): every panel imports, has `compute` / `draw` and a description, and all flags can be declared together.
5. Verify for real: `python analyze.py pretrain --run <backbone> --panel <name>` on outputs already on disk, look at the files, then
   `--redraw` and check the figure is byte-identical.
6. Presets: tag the panel in its own `PRESETS`; nothing else lists it. Add a line to the command block of `CLAUDE.md` only when the
   command differs from the plain `--panel <name>`.
