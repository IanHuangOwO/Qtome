"""Qtome's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import matplotlib
import torch

from model.Qtome.Qtome import QtomePretrain, build_finetune
from model.Qtome.Qtome_codebook_checker import QtomeCodebookChecker
from model.base_trainer import BaseTrainer
from model.base_plotter import BasePlotter
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


class QtomePlotter(BasePlotter):
    def plot_pretrain(self, filename='training_dashboard.png'):
        # Grouped: loss/reconstruction -> FFN routing -> architecture diagnostics. Order is the only grouping lever
        # `render`'s flat ncols grid gives us — no row breaks/section labels, so panels of a
        # group may still straddle a row edge.
        recon = ('masked', 'crimson'), ('unmasked', 'steelblue'), ('mse_patch', 'darkorchid')
        loss_panels = [
            dict(title="Total Loss (the training objective)\n(weighted sum of this run's loss terms: not comparable "
                       "across loss configs)", ylabel='Loss', series=[dict(key='loss', color='b')]),
            # Val is the comparable number (no dropout). masked is a 1.0 placeholder in the tokenizer
            # phase; mse_patch mixes masked and visible.
            dict(title='Recon MSE, train (solid) and val (dashed): compare runs on val\n(masked = 1.0 placeholder in '
                       'tokenizer phase; dropout puts train above val)',
                 ylabel='MSE', series=[dict(key=k, color=c) for k, c in recon]),
        ]
        nested = sorted((k for k in self.history['val'] if k.startswith('mse_nested_')), key=lambda k: int(k.rsplit('_', 1)[1]))
        for i, k in enumerate(nested):   # nested loss: error from the strongest k Q-atoms per patch
            loss_panels[1]['series'].append(dict(key=k, color=matplotlib.colormaps['cividis'](i / max(len(nested) - 1, 1)),
                                                 label=f"strongest {k.rsplit('_', 1)[1]}"))
        bands = [k for k in ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high') if self.has_signal(f'mse_band_{k}')]
        if bands:   # Qtome._band_errors: error power / target power per band, all valid positions
            cmap = matplotlib.colormaps['viridis']
            loss_panels.append(dict(title='Recon error per band, train (solid) and val (dashed)\n(error power / target '
                                          'power; 0.5-4 / 4-8 / 8-13 / 13-30 / 30-45 / 45-90 Hz)',
                                    ylabel='relative error (log)', logy=True,
                                    series=[dict(key=f'mse_band_{k}', color=cmap(i / max(len(bands) - 1, 1)), label=k)
                                            for i, k in enumerate(bands)]))

        # FFN Router Health — the MoEFFN routers inside every TSABlock (averaged across blocks),
        # see Qtome.update_ffn_router_metrics.
        ffn_router_series, ffn_twin_series = self.router_health_series(
            'ffn', entropy_label='Router entropy (load balance)')

        routing_panels = [
            dict(title='FFN Router Health\n(entropy rising = healthy spread; falling = collapse)',
                 ylabel='Entropy (higher=balanced)', series=ffn_router_series,
                 twin=dict(ylabel='Load std / LB loss', series=ffn_twin_series) if ffn_twin_series else None),
        ]

        architecture_panels = [
            dict(title='Per-Block Contribution Norm\n(flat near-zero = block not used)',
                 ylabel='Mean |delta| per block', series=self.indexed_series('block_norm_', cmap_name='viridis')),
        ]

        panels = loss_panels + routing_panels + architecture_panels
        self.render(panels, filename, suptitle='Tokenizer (Atom) Training Dashboard', ncols=4)

    def plot_finetune(self, filename='training_dashboard.png'):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Accuracy', ylabel='Acc', series=[dict(key='acc', color='crimson')]),
            dict(title='F1 (macro)', ylabel='F1', series=[dict(key='f1', color='steelblue')]),
            dict(title='F1 (weighted)', ylabel='F1', series=[dict(key='f1_weighted', color='teal')]),
            dict(title='Balanced Accuracy', ylabel='Bal. Acc', series=[dict(key='balanced_acc', color='darkorchid')]),
            dict(title="Cohen's Kappa", ylabel='Kappa', series=[dict(key='kappa', color='seagreen')]),
        ]
        self.render(panels, filename, suptitle='Training Dashboard')


PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=build_finetune,
    trainer_cls=QtomeTrainer,
    plotter_cls=QtomePlotter,
    codebook_checker_cls=QtomeCodebookChecker,
)
