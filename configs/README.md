# configs/

Grouped by owner: shared configs at the top, model-specific ones in `configs/<Model>/`.

| File / folder | Owner | What |
|---|---|---|
| `compile.json` | shared | Compile settings and the dataset list (`cache_dataset.py`). |
| `montages.json` | shared | Named channel montages: the canonical one and the sub-montages channel subsampling uses. |
| `finetune_protocols.json` | shared | Evaluation protocols: name -> split block (`IO/splits.py`), plus `_dataset_split` per-dataset settings (Compass sessions, few-shot fractions). Every model's cells use them. |
| `Qlet/pretrain.template.json` | Qlet | Starting point for a backbone's pretrain config: the corpus and the default recipe. One template for every corpus size (`window_fraction` sets the size). |
| `Qlet/finetune.template.json` | Qlet | Starting point for a finetune cell (a complete config). |
| `Qlet/protocol_heads.json` | Qlet | Qlet's part of each protocol: head, optimiser and fit, applied after the shared split. |
| `Qlet/sweeps/` | Qlet | Sweep files for `tools/misc/sweep.py` (e.g. `frozen_protocols.json`: every protocol x reported dataset x backbone). |
| `EEGNet/settings.json` | EEGNet | Training and preprocessing settings per split mode for `train_baseline.py` (Compass's, per-dataset overrides marked published / ours). |
| `runs/` | - | Real run configs, one folder per backbone -- **git-ignored**. |

## Corpus size

The pretraining corpus list (`dataset_params.pretrain`, written by `tools/misc/build_pretrain_corpus.py`) is the same at
every size. A size is one value, `preprocess_params.window_fraction`, the share of every subject's windows kept:
tiny 0.05 / small 0.2 / medium 0.5 / large 1.0 (the template's value). At one `window_fraction_seed` a smaller corpus
is a subset of a larger one.

## runs/

A real run never points at a template. `runs/` is git-ignored because every run snapshots its effective
config (overrides applied) into `output/<model>/<run>/artifacts/config.json`; those snapshots are the record.

```bash
cp configs/Qlet/pretrain.template.json configs/runs/<backbone>/pretrain.json    # set window_fraction for the size
cp configs/Qlet/finetune.template.json configs/runs/<backbone>/finetune/<head>/<cell>.json
# edit in the copy: pretrained_checkpoint, output_path, protocol, dataset_params.finetune
```

A run config is one complete JSON file (`load_config`, `IO/config.py`, only reads it). `model_name` is an identity string; `output_path` is where the run writes under `output/<model_type>/` (Qlet: `output/Qlet/`)
(pretrain default `<model_name>/pretrain`; finetune e.g. `<backbone>/finetune/<head>/<cell>`). A trained
backbone is always rebuilt from its checkpoint's own `build_config`, so editing `pretrain.json` afterwards
cannot change what a finetune loads -- but keep it as the record of how the backbone was trained anyway.
