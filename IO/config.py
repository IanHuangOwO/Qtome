"""Config loading and run-output paths shared by every entrypoint (train_*.py, cache_feature.py, analyze.py): overlay merge
(`base_config`), `--set` overrides, and where a run's folders live under output/."""
import copy
import json
import os


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(path: str) -> dict:
    """Load config. If it contains 'base_config', deep-merge on top of it."""
    with open(path, 'r') as f:
        cfg = json.load(f)
    if 'base_config' in cfg:
        base_path = cfg.pop('base_config')
        with open(base_path, 'r') as f:
            base = json.load(f)
        cfg = deep_merge(base, cfg)
    return cfg


def apply_overrides(cfg: dict, sets) -> dict:
    """In place: each 'dotted.key.path=value' in sets. value is parsed as JSON (numbers, true,
    null, lists, objects) and falls back to a plain string; a numeric path segment indexes a
    list (model_params.Qtome.finetune.features.0.time_rank=1). Missing dict keys are created,
    so a typo adds a key rather than failing -- check the config.json snapshot a run writes."""
    for item in sets or []:
        key, sep, raw = item.partition('=')
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node, parts = cfg, key.split('.')
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return cfg


QTOME_OUTPUT = 'output/Qtome'   # every Qtome run (backbones, their finetune heads, archive); tools resolve backbones here


def model_output_root(config: dict, mode: str = 'pretrain') -> str:
    """output/<model_type>: each model's runs live in their own folder (output/Qtome/, output/EEGNet/, ...)."""
    return os.path.join('output', config['training_params'][mode].get('model_type', 'Qtome'))


def resolve_output_path(config: dict, mode: str = 'pretrain') -> str:
    """Return this run's path segment under output/ -- training_params.<mode>.output_path
    if set, else '<model_name>/pretrain' for a pretrain run (pretrain artifacts always
    nest under pretrain/, so a backbone's later finetune/ runs never share a level with
    them) or plain model_name for a finetune run. output_path is separate from
    model_name (a clean identity string, e.g. logged in train_pretrain.py) so a path can
    carry a subfolder without model_name itself doing double duty as a path."""
    tp = config['training_params'][mode]
    if 'output_path' in tp:
        return tp['output_path']
    return f"{tp['model_name']}/pretrain" if mode == 'pretrain' else tp['model_name']


def resolve_output_dir(config: dict, *sub_dirs: str, mode: str = 'pretrain') -> str:
    """Return output/<model_type>/{output_path}/{sub_dirs...} and create it."""
    path = os.path.join(model_output_root(config, mode), resolve_output_path(config, mode=mode), *sub_dirs)
    os.makedirs(path, exist_ok=True)
    return path
