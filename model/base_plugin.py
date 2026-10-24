"""
BasePlugin: the bundle a model dir (model/<Name>/plugin.py) builds at module bottom and
registers in model/factory.py's MODEL_REGISTRY.
"""

from dataclasses import dataclass
from typing import Callable, Type

from model.base_trainer import BaseTrainer


@dataclass(frozen=True)
class BasePlugin:
    """One instance per model_type, built at the bottom of model/<Name>/plugin.py and
    registered in model/factory.py's MODEL_REGISTRY. Bundles everything a new model needs
    to plug into the shared train_pretrain.py/train_finetune.py loops."""
    build: Callable            # (bp: dict, num_channels: int) -> nn.Module, the pretrain backbone
    finetune_cls: Callable     # (backbone, num_channels, num_classes, **finetune params) -> classifier
    trainer_cls: Type[BaseTrainer]
