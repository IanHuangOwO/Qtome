"""The training-curve dashboard: rendered generically from a declarative list of panel specs (Dashboard.render), with the Qtome
pretrain and finetune layouts (QtomeDashboard). Shared by the pretrain and finetune training_dashboard panels."""

import os
import math

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def _ep(series):
    return range(1, len(series) + 1)


class Dashboard:
    """Generic training-curve dashboard renderer. Subclasses build a list of panel
    specs and call self.render(panels, filename, suptitle).

    Panel spec: {title, ylabel, series: [SeriesSpec, ...], twin: {ylabel, series} | None}
    SeriesSpec: {
      key: history key (looked up in both self.history['train'] and ['val']), or None
           if using override_train/override_val directly,
      label: legend label (default = key),
      color, style_train='-', style_val='--',
      band: bool — shade key+'_std' as a +/-1 std fill around the line,
      train_only: bool — skip the val line even if present,
      val_only: bool — skip the train line even if present,
      override_train / override_val: explicit value list, bypasses history lookup
    }
    A panel with no plottable series is left blank (axis off) instead of an empty frame.
    """

    def __init__(self, output_dir='output/visualization/training_curves', history=None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = history if history is not None else {'train': {}, 'val': {}}

    # -- helpers for building panel specs --------------------------------------

    def router_health_series(self, prefix='', entropy_label='Router entropy (higher=balanced)'):
        """MoE router-health panel shape shared by every router in this codebase (Expert,
        Q-atom, FFN, ...): router_entropy + gate_entropy on the main axis, router_load_std
        + lb_loss on a twin axis. History keys are f'{prefix}_<key>' (e.g. 'ffn_router_entropy',
        'ffn_lb_loss'), unprefixed for prefix=''.
        Returns (main_series, twin_series) — pass twin_series into a panel's twin dict
        only if non-empty, same guard every call site already used by hand."""
        def k(key):
            return f'{prefix}_{key}' if prefix else key

        main = [(k('router_entropy'), 'teal', entropy_label),
                (k('gate_entropy'), 'darkgoldenrod', 'Gate entropy (softmax weight)')]
        twin = [(k('router_load_std'), 'salmon', '--', 'Load std'),
                (k('lb_loss'), 'peru', ':', 'LB loss (1.0=uniform)')]

        main_series = [dict(key=key, color=c, label=l, train_only=True)
                       for key, c, l in main if key in self.history['train']]
        twin_series = [dict(key=key, color=c, style_train=ls, label=l, train_only=True)
                       for key, c, ls, l in twin if key in self.history['train']]
        return main_series, twin_series

    def indexed_series(self, prefix, cmap_name=None, train_only=True):
        """Unbounded family of series sharing a key prefix, sorted by trailing index
        (e.g. block_norm_0, block_norm_1, ...). train_only=False also plots the val curve
        for each key, when val history for that key exists."""
        hist = self.history['train']
        keys = sorted((k for k in hist if k.startswith(prefix)),
                      key=lambda k: int(k.rsplit('_', 1)[1]))
        n = len(keys)
        cmap = plt.get_cmap(cmap_name) if cmap_name else None
        series = []
        for i, k in enumerate(keys):
            color = cmap(i / max(n - 1, 1)) if cmap else f'C{i % 10}'
            series.append(dict(key=k, color=color, label=k, train_only=train_only))
        return series

    def has_signal(self, *keys):
        """True if any of keys has a finite nonzero value in train or val history."""
        return any(v == v and v != 0 for src in self.history.values() for k in keys for v in src.get(k, []))

    # -- rendering ---------------------------------------------------------------

    def render(self, panels, filename, suptitle=None, ncols=3):
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return

        n = len(panels)
        ncols = min(ncols, n) or 1
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), constrained_layout=True)
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).flatten()

        for ax, panel in zip(axes, panels):
            self._render_panel(ax, panel)
        for ax in axes[n:]:
            ax.axis('off')

        fig.savefig(os.path.join(self.output_dir, filename), dpi=110, bbox_inches='tight')
        plt.close(fig)

    def _render_panel(self, ax, panel):
        plotted = False
        for s in panel.get('series', []):
            plotted |= self._plot_one_series(ax, s)

        twin = panel.get('twin')
        if twin:
            ax2 = ax.twinx()
            twin_plotted = False
            for s in twin.get('series', []):
                twin_plotted |= self._plot_one_series(ax2, s)
            if twin_plotted:
                ax2.set_ylabel(twin.get('ylabel', ''))
                self._merge_legends(ax, ax2)
            plotted = plotted or twin_plotted
        elif plotted:
            ax.legend(fontsize='x-small')

        if plotted:
            if panel.get('logy'):
                ax.set_yscale('log')
            ax.set_title(panel.get('title', ''))
            ax.set_ylabel(panel.get('ylabel', ''))
            ax.set_xlabel('Epoch')
            ax.grid(True)
        else:
            ax.axis('off')

    def _plot_one_series(self, ax, s):
        key = s.get('key')
        tr = None if s.get('val_only') else (
            s.get('override_train') if s.get('override_train') is not None
            else (self.history['train'].get(key) if key else None))
        va = None if s.get('train_only') else (
            s.get('override_val') if s.get('override_val') is not None
            else (self.history['val'].get(key) if key else None))
        color = s.get('color', 'C0')
        label = s.get('label', key)
        plotted = False

        if tr:
            ax.plot(_ep(tr), tr, color=color, ls=s.get('style_train', '-'),
                    label=f"{label} (train)" if va else label)
            plotted = True
            if s.get('band') and key:
                std = self.history['train'].get(f'{key}_std')
                if std:
                    lo = [m - sd for m, sd in zip(tr, std)]
                    hi = [m + sd for m, sd in zip(tr, std)]
                    ax.fill_between(_ep(tr), lo, hi, color=color, alpha=0.15)

        if va:
            ax.plot(_ep(va), va, color=color, ls=s.get('style_val', '--'), alpha=0.7,
                    label=f"{label} (val)")
            plotted = True
            if s.get('band') and key:
                std = self.history['val'].get(f'{key}_std')
                if std:
                    lo = [m - sd for m, sd in zip(va, std)]
                    hi = [m + sd for m, sd in zip(va, std)]
                    ax.fill_between(_ep(va), lo, hi, color=color, alpha=0.08)

        return plotted

    @staticmethod
    def _merge_legends(ax1, ax2):
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, n1 + n2, loc='upper left', fontsize='x-small')

    def plot_pretrain(self, filename='training_dashboard.png'):
        raise NotImplementedError

    def plot_finetune(self, filename='training_dashboard.png'):
        raise NotImplementedError


class QtomeDashboard(Dashboard):
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
