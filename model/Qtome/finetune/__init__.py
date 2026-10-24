"""Qtome finetuning: head sources, protocol, fit, and the Q-atom code cache."""
import torch

# TF32 matmuls: training has always run with this (train_finetune.py set it on import, and the panels that imported train_finetune
# inherited it), and the feature cache is built with it. Set here so every process that touches the cache builds the same bits.
torch.set_float32_matmul_precision('high')
