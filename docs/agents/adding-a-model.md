# Adding a new model

**Two kinds of model live under `model/`.** A *pretrained backbone* (Qtome) follows the plugin protocol below and is registered in
`model/factory.py`. A *from-scratch baseline* (EEGNet) has no backbone and no plugin: put the architecture in `model/<Name>/<Name>.py`
(citation at the top, an assert-based `_selfcheck()`), register the class in `train_baseline.py`'s `BASELINES`, and give it
`configs/<Name>/settings.json` (training and preprocessing settings per split mode, per-dataset overrides marked published or ours).
`train_baseline.py` runs it through the same caches, protocol splits (`IO/splits.py` `make_runs`) and `group_eval.json` format as a
Qtome cell. Feed it microvolts: the caches store volts.

## A pretrained backbone

Read `docs/adr/0004-model-plugin-base-classes.md` first. **The rule:** a new backbone touches only files under `model/<Name>/`, one
line in `model/factory.py`, and its own panels. If you find yourself editing `train_pretrain.py`, `model/pretrain_loop.py` or
`model/base_trainer.py`, stop: either the model needs a real contract change (discuss first) or it is being solved the wrong way.
Qtome is the only model today, so this protocol has been exercised once; correct it when a second backbone is added.

What the shared training loop (`train_pretrain.py` + `model/pretrain_loop.py`) needs from a model:

### 1. The backbone: `model/<Name>/<Name>.py`

```python
class NamePretrain(nn.Module):
    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """x [B, C, N, L] patches, coords [B, C, 3], bool_masked_pos [B, C, N] (None = unmasked), valid_channels [B, C].
        Returns a SimpleNamespace with at least `recon` [B, C, N, L]; plus whatever the trainer's loss and metrics read."""

    def enter_tokenizer_phase(self): ...       # called once at the start of train_pretrain.py (unmasked phase)
    def enter_masked_phase(self, freeze_atoms=True): ...   # at the first masked epoch (epoch 1 when tokenizer_epochs = 0)
```

The phase is a persisted buffer (Qtome: `masked_phase`) restored by a `load_state_dict` hook, so a checkpoint loaded for analysis or
finetuning comes back in the phase it was saved in. `time_idx` is passed but Qtome ignores it. Give the model `patch_len` and
`patch_stride` attributes if the shared panels are to work on it.

### 2. The plugin: `model/<Name>/plugin.py`

```python
def build_model(bp, num_channels):
    """bp = config['model_params'][<Name>]['pretrain'] (read with bp.get defaults) -> NamePretrain."""

class NameTrainer(BaseTrainer):                      # model/base_trainer.py
    def compute_loss(self, model, x, out, mp, **hparams): ...   # -> (l_total, l_masked, l_unmasked); hparams = the config's `loss` block
    def epoch_metrics(self, model, out): ...                    # dict merged into the epoch's logged metrics
    def update_diagnostics(self, model, out): ...               # optional: per-step EMA state (default no-op)
    def epoch_snapshot(self, model): ...                        # optional: small arrays stacked over epochs into epoch_snapshots.npz

PLUGIN = BasePlugin(build=build_model, trainer_cls=NameTrainer)   # model/base_plugin.py
```

`model/Qtome/plugin.py` is the reference example.

### 3. Register: `model/factory.py`

```python
from model.Name.plugin import PLUGIN as NAME_PLUGIN
MODEL_REGISTRY = {'Qtome': QTOME_PLUGIN, 'Name': NAME_PLUGIN}
```

A pretrain checkpoint stores its `build_config` (`checkpoint_build_config`); `build_from_checkpoint` rebuilds the backbone from it, never from
an editable config, so an architecture change does not break old checkpoints as long as `build_model` still reads their keys.

### 4. Config

Add `model_params.<Name>.pretrain` (architecture, and a `loss` block whose keys reach `compute_loss` as `**hparams`) and set
`training_params.pretrain.model_type` to `"<Name>"`. Copy `configs/Qtome/pretrain.template.json` for the rest (corpus, masking, optimiser).

### 5. Finetuning and analysis

- **Finetuning is Qtome's.** `train_finetune.py` freezes a backbone and trains a `FeatureHead` on its Q-atom codes (`model/Qtome/head/`,
  `model/Qtome/finetune/`). A backbone with a different readout needs its own head code and entry point: there is no generic finetune hook.
- **Analysis is panels, not plugin hooks** (`docs/agents/adding-a-panel.md`). The snapshot panels call `model(...)` generically, so they work
  as they are; the codebook and the dashboard layouts read Qtome's `encode_atoms` and history keys, so a model with another readout
  adds its own panel files beside them.

### 6. Verify

There is no test suite; verify by hand.

```bash
# the model builds from a config
python -c "
import json
from model.factory import build_pretrain_from_config
cfg = json.load(open('configs/Qtome/pretrain.template.json'))
cfg['training_params']['pretrain']['model_type'] = 'Name'
m = build_pretrain_from_config(cfg); print(type(m).__name__, sum(p.numel() for p in m.parameters()))"

# a short real run exercises compute_loss, epoch_metrics and the phase switch end to end
python train_pretrain.py --config <a config with model_type=Name> --set training_params.pretrain.epochs=2 --set preprocess_params.window_fraction=0.01

# the stored history draws
python analyze.py pretrain --run <run> --panel training_dashboard
```
