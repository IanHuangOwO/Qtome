"""Gradients of the backbone on fixed probe batches, for the cross-task analyses (gradient_correlation, subspace_affinity,
pretrain_alignment; EEG-FM-Bench Figs. 3, 4 and 8, docs/references/2508.17742_EEG-FM-Bench.pdf).

For every task (a finetune cell of the group's backbone under --head: its dataset and one trained head) and every probe batch, two
gradients with respect to the backbone's parameters are taken on the SAME trials:
  ce   the classification loss of the trained head (the backbone is frozen in our finetunes, but the loss still has a gradient
       through it: trial -> backbone -> Q-atom codes -> head -> cross-entropy), computed as the head was trained (impute
       cells feed missing channels as mask tokens, as the atom cache does);
  rec  the pretraining loss (nested reconstruction, masks drawn per window from the four pretraining mask modes, real channels only).
Each gradient is split into parameter groups (GROUPS), compressed with a CountSketch to --sketch-dim values and L2-normalised, as in
the paper. Result: arrays [task, seed, batch, group, sketch_dim] for 'ce' and 'rec', cached in the numbers folder."""
import json
import os

import numpy as np
import torch

from IO.channels import get_standard_coords
from IO.dataset import build_dataset_from_config
from IO.preprocessing import slice_patches
from model.Qtome.finetune.feature_cache import subjects_of
from model.Qtome.atom_bank import AtomBank
from model.factory import load_finetune_checkpoint
from panels.core.numbers import load_npz, save_npz
from panels.shared.eval_windows import make_mask
from panels.shared.finetune_runs import cell_names, cell_run_dirs, head_files

GROUPS = ('patch_embed', 'coord_embed', 'temporal_attn', 'spatial_attn', 'ffn', 'norm', 'atoms')
MASK_KINDS = ('random_channel', 'channel_cluster', 'token_runs', 'time_block')   # the pretraining mask modes, one per window in turn


def group_of(name):
    """Parameter name -> group. Position embedding of the paper = coord_embed (FoPE has no parameters, so time position is not a group)."""
    if name.startswith(('embed.coord_proj', 'spatial_bias')):
        return 'coord_embed'
    if name.startswith(('embed.', 'mask_token')):
        return 'patch_embed'
    if name.startswith('atoms.'):
        return 'atoms'
    if '.temporal_attn.' in name:
        return 'temporal_attn'
    if '.spatial_attn.' in name:
        return 'spatial_attn'
    if '.ffn.' in name:
        return 'ffn'
    if '.norm' in name or '.scale_' in name:
        return 'norm'
    raise ValueError(f'parameter {name!r} has no group: add it to panels/shared/gradient_probe.py group_of')


class CountSketch:
    """Linear sketch of a vector of n values to dim values: out[h(i)] += s(i) x[i], with a fixed seeded hash h and sign s, the same for
    every gradient of a group (so cosines between gradients are comparable)."""
    def __init__(self, n, dim, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.h = torch.randint(0, dim, (n,), generator=g).to(device)
        self.s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)
        self.dim = dim

    def __call__(self, v):
        return torch.zeros(self.dim, device=v.device).index_add_(0, self.h, v * self.s)


class Task:
    """One finetune cell: the trained head and a pool of its dataset's trials, ready to give probe batches."""

    def __init__(self, bb, head, cell, device, n_subjects, fold=0):
        run_dirs = cell_run_dirs(bb, head, cell)
        self.cell, heads = cell, head_files(run_dirs)
        if not heads:
            raise FileNotFoundError(f'no trained head under {bb}/finetune/{head}/{cell}')
        config = json.load(open(os.path.join(run_dirs[0], 'artifacts', 'config.json')))
        (self.dataset, ds_args), = config['dataset_params']['finetune'].items()
        self.impute = bool(ds_args.get('impute_missing_channels'))
        self.model = load_finetune_checkpoint(config, heads[fold % len(heads)], device)   # backbone + one fold's head
        if self.model.extractor is None:
            raise ValueError(f'{cell}: the head reads no Q-atom code (raw-input head): nothing to take a backbone gradient through')
        self.device = device
        subs = subjects_of(ds_args)[:n_subjects]
        cfg = json.loads(json.dumps(config))
        cfg['dataset_params']['finetune'] = {self.dataset: {**ds_args, 'subject_to_use': subs}}
        base = build_dataset_from_config(cfg, mode='finetune').base_dataset
        pp = config['preprocess_params']
        self.patch_len = pp.get('patch_length', 100)
        self.patch_stride = pp.get('patch_stride', self.patch_len)
        self.x, self.labels = base.data, base.labels.long()
        self.valid = base.all_valid_channels[0]
        assert all(torch.equal(v, self.valid) for v in base.all_valid_channels), 'one real-channel set per dataset'
        coords = base.all_coords[0].clone()
        self.coords_real = coords.clone()
        if self.impute:      # missing channels at their standard 10-10 position, as the atom cache does
            for i in torch.nonzero(~self.valid).flatten().tolist():
                coords[i] = torch.as_tensor(get_standard_coords(base.channel_names[i]), dtype=coords.dtype)
        self.coords = coords

    def batch(self, idx):
        """Trials idx -> dict of device tensors: patches [b, C, N, L], valid [b, C], coords for the head path and the real ones."""
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)
        b, n = len(idx), xp.shape[2]
        dev = self.device
        return dict(x=xp.to(dev), y=self.labels[idx].to(dev), t=torch.arange(n, device=dev).unsqueeze(0).expand(b, n),
                    valid=self.valid.unsqueeze(0).expand(b, -1).to(dev), idx=idx,
                    coords=self.coords.unsqueeze(0).expand(b, -1, -1).to(dev),
                    coords_real=self.coords_real.unsqueeze(0).expand(b, -1, -1).to(dev))

    def ce_loss(self, b):
        """Cross-entropy of the trained head on the Q-atom code, with the gradient taken through the backbone."""
        amp = atom_code(self.model.extractor, b['x'], b['coords'], b['t'], b['valid'], self.impute)
        return torch.nn.functional.cross_entropy(self.model.head({'atom': amp}), b['y'])

    def rec_loss(self, b):
        """The pretraining loss on the same trials: one mask per window (the four pretraining modes in turn, seeded by trial index),
        real channels only."""
        x, valid = b['x'], b['valid']
        B, C, N, _ = x.shape
        real_patch = x.abs().amax(dim=(1, 3)) > 0                                           # [B, N]
        mask = torch.zeros(B, C, N, dtype=torch.bool)
        for j in range(B):
            tok = (valid[j].cpu()[:, None] & real_patch[j].cpu()[None, :])
            m, _ = make_mask(MASK_KINDS[j % len(MASK_KINDS)], tok, b['coords_real'][j].cpu(), int(b['idx'][j]), None)
            mask[j] = m & tok
        mask = mask.to(x.device)
        bb = self.model.backbone
        out = bb(x, b['coords_real'], b['t'], bool_masked_pos=mask, valid_channels=valid)
        loss, _, _ = bb.get_loss(x, out.recon, out.ranked, mask, ffn_lb_loss=out.ffn_lb_loss, valid_channels=valid)
        return loss


def atom_code(extractor, x, coords, time_idx, valid, impute):
    """AtomExtractor.forward (model/Qtome/head/extractor.py) with gradients: the same computation, but that forward and
    AtomBank.dense_amp are both @torch.no_grad, so the body is repeated here (check: panels/shared/gradient_probe._selfcheck)."""
    bb = extractor.backbone
    B, C, N, _ = x.shape
    rms = x.float().pow(2).mean(-1).sqrt()                                                  # [B, C, N]
    vmask, masked = valid, None
    if impute:
        miss = ~valid
        masked = miss[:, :, None].expand(B, C, N)
        d = torch.cdist(coords.float(), coords.float()).masked_fill(miss[:, None, :], float('inf'))
        nn_idx = d.topk(min(3, int(valid.sum(1).min())), dim=-1, largest=False).indices
        nn_rms = torch.gather(rms[:, None].expand(B, C, C, N), 2, nn_idx[..., None].expand(-1, -1, -1, N)).mean(2)
        rms = torch.where(miss[:, :, None], nn_rms, rms)
        vmask = torch.ones_like(valid)
    z, _ = bb.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=masked, valid_channels=vmask)
    zg = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
    amp = AtomBank.dense_amp.__wrapped__(bb.atoms, zg, rms=rms.permute(0, 2, 1).reshape(B * N, C, 1))
    amp = amp.reshape(B, N, C, -1, 2).float() * vmask.float()[:, None, :, None, None]
    return amp[:, :, extractor.channel_idx]


def sketched_gradient(loss, params, sketches):
    """Gradient of loss w.r.t. every parameter -> [n_groups, dim]: each group's gradients concatenated, sketched, L2-normalised
    (a group without gradient gives NaN)."""
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    out = []
    for g in GROUPS:
        parts = [(torch.zeros_like(p) if gr is None else gr).flatten() for (n, p), gr in zip(params, grads) if group_of(n) == g]
        v = sketches[g](torch.cat(parts))
        norm = v.norm()
        out.append(v / norm if norm > 0 else torch.full_like(v, float('nan')))
    return torch.stack(out)


def collect(bb, head, cells, device, n_seeds=5, n_batches=16, batch_size=16, sketch_dim=1024, n_subjects=3, fold=0, log=print):
    """-> dict(ce, rec [task, seed, batch, group, dim] float32, tasks [dataset names], cells, groups). Probe batches are drawn
    at random without replacement from the task's trial pool, seeded by (seed, task order)."""
    ce, rec, names = [], [], []
    for ti, cell in enumerate(cells):
        task = Task(bb, head, cell, device, n_subjects, fold)
        task.model.eval()
        params = [(n, p) for n, p in task.model.backbone.named_parameters()]
        for _, p in params:
            p.requires_grad_(True)
        sketches = {g: CountSketch(sum(p.numel() for n, p in params if group_of(n) == g), sketch_dim, 1000 + i, device)
                    for i, g in enumerate(GROUPS)}
        names.append(task.dataset)
        c_t, r_t = [], []
        for seed in range(n_seeds):
            perm = torch.randperm(len(task.x), generator=torch.Generator().manual_seed(seed * 1000 + ti))
            c_s, r_s = [], []
            for k in range(n_batches):
                b = task.batch(perm[(k * batch_size) % len(perm):][:batch_size].sort().values)
                c_s.append(sketched_gradient(task.ce_loss(b), params, sketches).cpu())
                r_s.append(sketched_gradient(task.rec_loss(b), params, sketches).cpu())
            c_t.append(torch.stack(c_s)); r_t.append(torch.stack(r_s))
            log(f'  {task.dataset}: seed {seed + 1}/{n_seeds} done')
        ce.append(torch.stack(c_t)); rec.append(torch.stack(r_t))
        del task, params
        torch.cuda.empty_cache()
    return dict(ce=torch.stack(ce).numpy(), rec=torch.stack(rec).numpy(), tasks=np.array(names), cells=np.array(cells),
                groups=np.array(GROUPS))


def flags(f):
    """The options of every panel that reads the probe (declared by each, identical defaults)."""
    f.add('--cells', nargs='+', help='finetune cells = tasks (default: the <dataset>_impute64_loso cells of the group, else every cell)')
    f.add('--probe-seeds', type=int, default=5, help='repeats with other probe batches (mean +- std over them)')
    f.add('--probe-batches', type=int, default=16, help='probe batches per task and repeat (rows of the subspace-affinity PCA)')
    f.add('--probe-batch-size', type=int, default=16, help='trials per probe batch')
    f.add('--sketch-dim', type=int, default=1024, help='CountSketch dimension of every gradient')
    f.add('--probe-fold', type=int, default=0, help='which fold\'s trained head every task uses (index into its run_* folders)')
    f.add('--probe-subjects', type=int, default=3, help='subjects of each dataset the probe trials are drawn from')


def default_cells(bb, head):
    names = cell_names([bb], head)
    imp = [n for n in names if '_impute64_loso' in n]
    return imp or names


def gradients(ctx, group):
    """The saved probe gradients of one group of the finetune run, computed on first use (settings recorded: other settings recompute)."""
    a = ctx.args
    bb = ctx.groups[group]
    cells = a.cells or default_cells(bb, ctx.head)
    settings = dict(bb=bb, head=ctx.head, cells=cells, seeds=a.probe_seeds, batches=a.probe_batches, batch_size=a.probe_batch_size,
                    sketch_dim=a.sketch_dim, subjects=a.probe_subjects, fold=a.probe_fold)
    path = ctx.numbers(f'gradient_probe_{group}.npz')
    cache = getattr(ctx, '_gradients', None)
    if cache is None:
        cache = ctx._gradients = {}
    if group not in cache:
        if os.path.exists(path) and json.loads(str(load_npz(path)['settings'])) == settings:
            cache[group] = load_npz(path)
        else:
            z = collect(bb, ctx.head, cells, ctx.device, a.probe_seeds, a.probe_batches, a.probe_batch_size, a.sketch_dim,
                        a.probe_subjects, a.probe_fold)
            save_npz(path, settings=np.array(json.dumps(settings)), **z)
            cache[group] = load_npz(path)
    return cache[group]


def _selfcheck(bb='qtome_smoke', head='allatom', cell='BNCI2014001_impute64_loso_seed1'):
    """atom_code equals AtomExtractor.forward (bit-identical values), and its loss has a gradient on every group."""
    task = Task(bb, head, cell, torch.device('cuda'), 1)
    b = task.batch(torch.arange(8))
    with torch.no_grad():
        ref = task.model.extractor(b['x'], b['coords'], b['t'], b['valid'], impute_missing=task.impute)
        assert torch.equal(ref, atom_code(task.model.extractor, b['x'], b['coords'], b['t'], b['valid'], task.impute))
    params = list(task.model.backbone.named_parameters())
    for _, p in params:
        p.requires_grad_(True)
    sk = {g: CountSketch(sum(p.numel() for n, p in params if group_of(n) == g), 64, 0, 'cuda') for g in GROUPS}
    assert torch.isfinite(sketched_gradient(task.ce_loss(b), params, sk)).all(), 'ce gradient: a group has none'
    assert torch.isfinite(sketched_gradient(task.rec_loss(b), params, sk)).all(), 'rec gradient: a group has none'
