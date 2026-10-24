import torch
from model.Qtome.plugin import PLUGIN as QTOME_PLUGIN
from IO.channels import resolve_canonical_channels

# Adding a model = implement model/<Name>/plugin.py (Trainer/Checker/Plotter + build_model,
# bundled into a BasePlugin) and register the PLUGIN instance here. No other shared file
# needs editing.
MODEL_REGISTRY = {
    'Qtome': QTOME_PLUGIN,
}


def checkpoint_build_config(config, mode='pretrain'):
    """Everything that decides the backbone's architecture, saved inside every pretrain checkpoint:
    a trained backbone is always rebuilt from its own checkpoint, never from a config file that
    may have been edited since it was trained."""
    model_type = config['training_params'][mode].get('model_type', 'Qtome')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    canonical_channels = config.get('preprocess_params', {}).get('canonical_channels')
    if not canonical_channels:
        raise ValueError("preprocess_params.canonical_channels must be set (it fixes the channel count)")
    return {'model_type': model_type, 'num_channels': len(resolve_canonical_channels(canonical_channels)),
            'model_params': config['model_params'][model_type]['pretrain']}


def build_pretrain_from_config(config, mode='pretrain'):
    """A fresh, untrained backbone -- training starts here; a trained one comes from build_from_checkpoint."""
    bc = checkpoint_build_config(config, mode)
    return MODEL_REGISTRY[bc['model_type']].build(bc['model_params'], bc['num_channels'])


def build_from_checkpoint(ckpt):
    """The backbone exactly as trained: built from ckpt['build_config'], weights loaded (the load
    restores its phase flags, Qtome _restore_phase)."""
    if 'build_config' not in ckpt:
        raise ValueError("checkpoint has no build_config: trained before the 2026-09-26 encoder redesign, "
                         "the current code cannot rebuild it")
    bc = ckpt['build_config']
    model = MODEL_REGISTRY[bc['model_type']].build(bc['model_params'], bc['num_channels'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def optimizer_param_groups(model, weight_decay):
    """Splits model.parameters() into decay/no-decay groups for AdamW — ndim<=1 params
    (LayerNorm/RMSNorm weights, every bias) get weight_decay=0.0, everything else (linear/
    conv/embedding matrices, AtomBank's W_down/W_out/u, ...) gets the configured decay.
    Standard BERT/ViT-style recipe: decaying a norm's gain or a bias toward zero fights the
    norm's job and has no overfitting-prevention upside (biases have no capacity to
    memorize on their own), so excluding them is a strict improvement, not a tunable
    tradeoff — unlike the decay *value* itself, which does trade fit against generalization."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def load_backbone(config, checkpoint_path=None, mode='finetune'):
    """Frozen-backbone loader shared by the finetune builders and the feature cache."""
    path = checkpoint_path or config['training_params'][mode]['pretrained_checkpoint']
    return build_from_checkpoint(torch.load(path, map_location='cpu', weights_only=False))


def load_finetune_checkpoint(config, path, device):
    """Rebuild a finetune model from a head checkpoint: backbone from ckpt['backbone_checkpoint'],
    head from ckpt['head_config'] (no shape inference)."""
    from model.Qtome.Qtome import FinetuneModel
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    backbone = load_backbone(config, ckpt['backbone_checkpoint'])
    return FinetuneModel.from_checkpoint(backbone, ckpt).to(device).eval()
