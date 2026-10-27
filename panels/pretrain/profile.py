"""profile: parameter counts and per-component forward timing of a fresh model built from the config (no checkpoint needed; without
--run the pretrain template).

Model profiling: parameter counts and per-component forward-pass timing, from a fresh
untrained model (configs/Qlet/pretrain.template.json's architecture) -- no checkpoint, no dataset. Moved
from the old standalone profile_model.py; the report is printed by compute().
Option: --train (train-mode timing)."""
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn

from model.factory import build_pretrain_from_config

logging.getLogger("fvcore").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProfilerHooks:
    def __init__(self):
        self.timings = defaultdict(list)
        self.starts = {}
        self.device = None

    def _sync(self):
        if self.device and self.device.type == 'cuda':
            torch.cuda.synchronize()

    def register(self, model, device):
        self.device = device
        hooks = []
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleList):
                for sub_module in module:
                    h1 = sub_module.register_forward_pre_hook(self._make_pre_hook(name))
                    h2 = sub_module.register_forward_hook(self._make_hook(name))
                    hooks.extend([h1, h2])
            else:
                h1 = module.register_forward_pre_hook(self._make_pre_hook(name))
                h2 = module.register_forward_hook(self._make_hook(name))
                hooks.extend([h1, h2])
        return hooks

    def _make_pre_hook(self, name):
        def hook(module, input):
            self._sync()
            self.starts[name] = time.perf_counter()
        return hook

    def _make_hook(self, name):
        def hook(module, input, output):
            self._sync()
            end = time.perf_counter()
            if name in self.starts:
                duration = (end - self.starts[name]) * 1000  # ms
                self.timings[name].append(duration)
        return hook

    def get_summary(self, n_iters, model):
        stats = []
        for name, module in model.named_children():
            times = self.timings.get(name, [])
            if not times:
                continue
            if isinstance(module, nn.ModuleList):
                total_ms = sum(times) / n_iters
            else:
                total_ms = sum(times) / len(times)
            stats.append((name, total_ms))
        return stats


@dataclass
class ProfileResult:
    model_type: str
    batch: int
    channels: int
    patches: int
    patch_len: int
    children: list      # [(name, module)] in named_children() order -- also print order
    param_map: dict      # name -> trainable param count
    time_stats: dict     # name -> avg forward-pass ms
    loss_ms: float
    total_ms: float


def run_profile(config, device, train_mode=False):
    """Builds a fresh (untrained) model from config, times its forward pass and get_loss
    component-by-component via forward hooks. No checkpoint, no dataset -- dummy input
    shaped from preprocess_params. Returns a ProfileResult; printing the report is the
    caller's (panel_profile.py's) job, not this function's."""
    model_type = config['training_params']['pretrain'].get('model_type', 'Qlet')
    preprocess = config['preprocess_params']

    B, C = 16, 64
    L = preprocess.get('patch_length', 25)
    N = 800 // L  # 4 seconds @ 200Hz = 800 samples

    model = build_pretrain_from_config(config).to(device)
    model.train() if train_mode else model.eval()

    x = torch.randn(B, C, N, L).to(device)
    coords = torch.randn(B, C, 3).to(device)
    time_idx = torch.zeros(B, N, dtype=torch.long).to(device)

    with torch.no_grad():
        model(x, coords, time_idx)

    children = list(model.named_children())
    param_map = {name: count_parameters(module) for name, module in children}

    profiler = ProfilerHooks()
    profiler.register(model, device)

    with torch.no_grad():
        for _ in range(5):
            model(x, coords, time_idx)
    profiler.timings.clear()

    n_iters = 20
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_total = time.perf_counter()

    loss_times = []
    bool_masked_pos = torch.zeros(B, C, N, dtype=torch.bool).to(device)
    with torch.no_grad():
        for _ in range(n_iters):
            out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model.get_loss(x, out.recon, out.ranked, bool_masked_pos=bool_masked_pos)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            loss_times.append((t2 - t1) * 1000)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_avg_ms = ((time.perf_counter() - start_total) / n_iters) * 1000
    time_stats = dict(profiler.get_summary(n_iters, model))
    avg_loss_ms = sum(loss_times) / n_iters

    return ProfileResult(
        model_type=model_type, batch=B, channels=C, patches=N, patch_len=L,
        children=children, param_map=param_map, time_stats=time_stats,
        loss_ms=avg_loss_ms, total_ms=total_avg_ms,
    )


NEEDS_CHECKPOINT = False


def add_args(flags):
    flags.add('--train', action='store_true', default=False, help='profile: train-mode timing')


def compute(ctx):
    r = run_profile(ctx.config, ctx.device, train_mode=ctx.args.train)
    print(f"Model {r.model_type} on {ctx.device}, {'train' if ctx.args.train else 'eval'} mode; input "
          f"B={r.batch} C={r.channels} N={r.patches} L={r.patch_len}")
    print(f"{'component':<22} | {'params (M)':<10} | {'time (ms)':<10} | % total")
    for name, _ in r.children:
        t, p = r.time_stats.get(name, 0.0), r.param_map.get(name, 0)
        print(f"{name:<22} | {p / 1e6:<10.2f} | {t:<10.2f} | {100 * t / r.total_ms:6.1f}%")
    print(f"{'get_loss':<22} | {'-':<10} | {r.loss_ms:<10.2f} | {100 * r.loss_ms / r.total_ms:6.1f}%")
    print(f"{'total':<22} | {sum(r.param_map.values()) / 1e6:<10.2f} | {r.total_ms:<10.2f} | 100.0%")
