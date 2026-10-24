"""task_filter_filter: the filter of --filter-source alone, applied to nothing, at --share: power kept of white input, effective rank and the
importance of each channel for each Q-atom. Numbers -> analysis/task_filter/<source>/filter_analysis_<share>.npz, figure ->
analysis/pretrain/task_filter/<source>/filter_analysis_<share>.png (drawn from the saved numbers). Builds the filter first if it is not saved."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from IO.loader import get_standard_coords
from model.Qtome.Qtome_filter_tools import CHANNELS
from panels.kit.topomap import draw_topomap, project_coords_2d
from panels.shared.task_filter.context import filter_context

PRESETS = set()


def add_args(flags):
    flags.add('--filter-source', default='multi', help="'multi' or a dataset name")
    flags.add('--filter-sources', nargs='+', help='the datasets whose heads build the multi filter (default: Qtome_filter_tools SOURCES)')
    flags.add('--share', type=float, default=0.1, choices=[0.5, 0.25, 0.1], help='share of cells kept')


def filter_numbers(ctx, share, path):
    """The numbers of the filter alone, saved to path (npz): per Q-atom dominant frequency, power kept of white input and
    effective rank of the filter at this share, and the importance [64, S] of each channel for each Q-atom."""
    mean_op = ctx.filt['filters'][share].numpy(); C = mean_op.shape[1]
    sv = np.linalg.svd(mean_op, compute_uv=False)
    np.savez(path, peak=np.asarray(ctx.backbone.freqs), white=(sv ** 2).sum(1) / C, rank=sv.sum(1) ** 2 / (sv ** 2).sum(1),
             importance=ctx.filt['importance'].numpy(), source=ctx.filt['source'], share=share)


def draw_filter(path, out_path, log=print):
    """The filter alone, whatever it is applied to, drawn from filter_numbers' file: per Q-atom the power it keeps of white
    input (equal power on every channel, no correlation), its effective rank (channel patterns kept), and IMPORTANCE of each
    channel (class information less chance from the heads that built it, as a share of the total over every channel and
    Q-atom; ONE colour scale, so a Q-atom that matters little is pale everywhere). Q-atoms most important first."""
    z = np.load(path); SRC, SHARE, peak, white, rank, imp = str(z['source']), float(z['share']), z['peak'], z['white'], z['rank'], z['importance']
    C, S = imp.shape; imp = imp / imp.sum(); ia = imp.sum(0); order = np.argsort(-ia)
    pos = project_coords_2d(np.array([get_standard_coords(n) for n in CHANNELS]))
    fig = plt.figure(figsize=(24, 27)); fl = fig.add_gridspec(2, 1, height_ratios=[.9, 6.2], left=.02, right=.98, top=.93, bottom=.035, hspace=.12)
    fo = fl[0].subgridspec(1, 2, wspace=.2); a_ = fig.add_subplot(fo[0]); a_.scatter(peak, white, c=white, cmap='YlOrRd', edgecolor='k', s=70, zorder=3)
    for i in range(S): a_.annotate(str(i), (peak[i], white[i]), fontsize=8, xytext=(4, 3), textcoords='offset points')
    a_.set_xscale('log'); a_.set_xticks([2, 4, 8, 13, 30, 60, 100]); a_.set_xticklabels([2, 4, 8, 13, 30, 60, 100]); a_.grid(alpha=.3); a_.set_ylim(0, None)
    a_.set_xlabel('dominant frequency of the Q-atom, Hz'); a_.set_ylabel('power kept'); a_.set_title('power kept of white input, per Q-atom')
    fq = np.argsort(peak, kind='stable')                                  # by frequency, as the scatter
    a_ = fig.add_subplot(fo[1]); a_.bar(np.arange(S), rank[fq], color=plt.cm.YlOrRd(ia[fq] / ia.max()), edgecolor='k', lw=.6)
    a_.set_xticks(np.arange(S)); a_.set_xticklabels([f'{i}\n{peak[i]:.0f}' for i in fq], fontsize=8); a_.set_ylim(0, C); a_.grid(axis='y', alpha=.3)
    a_.set_xlabel('Q-atom (index, Hz), by frequency; colour: importance'); a_.set_ylabel(f'of {C} channels'); a_.set_title('number of channel patterns kept (effective rank)')
    gm = fl[1].subgridspec(6, 4, hspace=.32, wspace=.2); sa = [fig.add_subplot(gm[i, j]) for i in range(6) for j in range(4)]
    for i, a_ in zip(order, sa):
        im = draw_topomap(a_, pos, imp[:, i], vmin=0, vmax=imp.max(), cmap='YlOrRd'); a_.set_title(f'{i}: {peak[i]:.0f} Hz, {ia[i]:.1%}', fontsize=9)
    fig.colorbar(im, ax=sa, shrink=.4, fraction=.02, pad=.01, location='bottom', label='share of the total importance').ax.tick_params(labelsize=8)
    fig.text(.5, .99, f'THE FILTER fitted on {SRC}, all 64 channels, share {SHARE:g}, time-free', ha='center', va='top', fontsize=20, weight='bold')
    xs_ = gm.get_grid_positions(fig); fig.text(.5, xs_[1][0] + .012, 'importance of each channel for each Q-atom: class information less chance (index: Hz, share of the total), most important first; one scale', ha='center', va='bottom', fontsize=13)
    fig.savefig(out_path, dpi=80); plt.close(fig)
    log(f'filter from {SRC}, share {SHARE:g}: {out_path}')


def compute(ctx):
    _, tf = filter_context(ctx, ctx.args.filter_source)
    src, share = ctx.args.filter_source, ctx.args.share
    filter_numbers(tf, share, ctx.numbers('task_filter', src, f'filter_analysis_{share:g}.npz'))


def draw(ctx):
    src, share = ctx.args.filter_source, ctx.args.share
    draw_filter(ctx.numbers('task_filter', src, f'filter_analysis_{share:g}.npz'),
                ctx.figure('task_filter', src, f'filter_analysis_{share:g}.png'))
