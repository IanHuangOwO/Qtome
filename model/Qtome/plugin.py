"""Qtome's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_plugin.py)."""

import torch

from model.Qtome.Qtome import QtomePretrain
from model.base_trainer import BaseTrainer
from model.base_plugin import BasePlugin


def build_model(bp, num_channels):
    """bp: config['model_params']['Qtome']['pretrain']."""
    sb = bp.get('atom_bank', {})
    moe_ffn = bp.get('moe_ffn', {})

    return QtomePretrain(
        embed_dim=bp.get('embed_dim', 100),
        enc_depth=bp.get('enc_depth', 12),
        mlp_ratio=moe_ffn.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 20),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        num_channels=num_channels,
        spatial_embedding=bp.get('spatial_embedding', True),
        n_atoms=sb.get('n_atoms', 16),
        atom_hidden_width=sb.get('hidden_width', 16),
        n_routed_ffn_experts=moe_ffn.get('n_routed_experts', 4),
        n_shared_ffn_experts=moe_ffn.get('n_shared_experts', 1),
        ffn_top_k=moe_ffn.get('top_k', 2),
        # patch_stride duplicates preprocess_params (same convention as patch_len
        # above) — the shared build_model(bp, num_channels) interface
        # (model/factory.py) doesn't pass preprocess_params through.
        patch_stride=bp.get('patch_stride'),
    )


class QtomeTrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, **hparams):
        ffn_lb_weight = hparams.get('ffn_lb_weight', 0.01)
        return model.get_loss(x, out.recon, out.ranked, bool_masked_pos=mp,
                               mse_patch_weight=hparams.get('mse_patch_weight', 1.0),
                               unmasked_weight=hparams.get('unmasked_weight', 1.0),
                               ffn_lb_loss=out.ffn_lb_loss, ffn_lb_weight=ffn_lb_weight,
                               valid_channels=out.valid_channels,
                               # nested reconstruction loss (docs/adr/0018); unset = QtomePretrain.NESTED_* defaults
                               nested_sizes=hparams.get('nested_sizes'), nested_weights=hparams.get('nested_weights'))

    @torch.no_grad()
    def epoch_snapshot(self, model):
        D, H = model.atoms.templates()                              # unit templates and quadrature partners [n_atoms, patch_len]
        return {'atom_templates': D.float().cpu().numpy(), 'atom_quadrature': H.float().cpu().numpy()}

    def update_diagnostics(self, model, out):
        model.update_ffn_router_metrics(out.ffn_router_entropy, out.ffn_router_load_std, out.ffn_gate_entropy)

    def epoch_metrics(self, model, out):
        # mse_patch (and the other mse_* terms) are accumulated per-batch and epoch-averaged in
        # train_pretrain.py (train_one_epoch/validate_one_epoch), not added here — this
        # function only ever sees the last batch's out, which would make them a
        # last-batch snapshot instead of an epoch average like every other loss stat.
        metrics = model.get_metrics()
        metrics['ffn_lb_loss'] = out.ffn_lb_loss.item() if hasattr(out.ffn_lb_loss, 'item') else float(out.ffn_lb_loss)
        return metrics


PLUGIN = BasePlugin(
    build=build_model,
    trainer_cls=QtomeTrainer,
)
