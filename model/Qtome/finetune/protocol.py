"""Protocol -> split and head: training_params.finetune.protocol picks the shared split and Qtome's head and optimiser for it."""
import json

from IO.config import apply_overrides
from IO.splits import protocol_split


QTOME_HEADS = 'configs/Qtome/protocol_heads.json'


def apply_protocol(config, heads=QTOME_HEADS):
    """training_params.finetune.protocol = <name>: the protocol's split (IO/splits.py protocol_split: the shared table
    plus per-dataset settings), then Qtome's head and optimiser settings for it (configs/Qtome/protocol_heads.json,
    dotted keys in order). No protocol key: unchanged."""
    name = config['training_params']['finetune'].get('protocol')
    if not name:
        return config
    split = protocol_split(name, list(config['dataset_params']['finetune']))
    table = {k: v for k, v in json.load(open(heads)).items() if not k.startswith('_')}
    if name not in table:
        raise ValueError(f"protocol {name!r} has no Qtome head in {heads}")
    return apply_overrides(config, [f'training_params.finetune.split={json.dumps(split)}'] +
                           [f'{k}={json.dumps(v)}' for k, v in table[name].items()])
