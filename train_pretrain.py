import os
import sys
import json
import argparse
import socket
import subprocess
import time
from datetime import datetime

import copy
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from IO.splits import split_pretrain_subjects
from IO.wrappers import MontageBatchSampler
from IO.masking import build_masking_strategy_from_config
from model.factory import build_pretrain_from_config, checkpoint_build_config, optimizer_param_groups, MODEL_REGISTRY
from model.history import History
from IO.config import apply_overrides, resolve_output_path, model_output_root, setup_logger
from model.pretrain_loop import train_one_epoch, unpack_batch, validate_one_epoch

torch.set_float32_matmul_precision('high')


def _run_info(config, model, train_dataset, val_dataset, device):
    """Provenance of a run, stored as artifacts/run_info.json (finished_at / status / epochs_done are filled at the end):
    code version, environment, hardware, model size, corpus size."""
    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    try:
        import mne
        mne_version = mne.__version__
    except ImportError:
        mne_version = None
    return {'started_at': datetime.now().isoformat(timespec='seconds'), 'host': socket.gethostname(),
            'git_commit': sh('git', 'rev-parse', 'HEAD'), 'git_dirty': bool(sh('git', 'status', '--porcelain')),
            'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'mne': mne_version,
            'gpu': torch.cuda.get_device_name(device) if 'cuda' in str(device) else None,
            'n_params': sum(p.numel() for p in model.parameters()),
            'n_train_windows': len(train_dataset), 'n_val_windows': len(val_dataset),
            'epochs': config['training_params']['pretrain']['epochs'], 'status': 'running'}


def main():
    parser = argparse.ArgumentParser(description='Qlet pretraining, one run, two phases: unmasked tokenizer phase '
                                                   '(training_params.pretrain.tokenizer_epochs), then masked phase.')
    parser.add_argument('--config', type=str, default='configs/Qlet/pretrain.template.json')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config value, dotted path, JSON value (repeatable), '
                             'e.g. --set training_params.pretrain.seed=2')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = apply_overrides(json.load(f), args.set)

    train_params = config['training_params']['pretrain']
    if train_params.get('num_threads'):   # CPU threads for this process (parallel runs share the cores)
        torch.set_num_threads(int(train_params['num_threads']))
    device     = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_params.get('model_name', 'default_run')
    train_params.setdefault('model_name', model_name)

    base_output_dir = os.path.join(model_output_root(config, 'pretrain'), resolve_output_path(config, mode='pretrain'))
    checkpoint_dir  = os.path.join(base_output_dir, "checkpoint")
    artifact_dir    = os.path.join(base_output_dir, "artifacts")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    logger, timestamp = setup_logger(artifact_dir)
    for name in ('config.json', f'config_{timestamp}.json'):     # the effective config, overrides applied
        with open(os.path.join(artifact_dir, name), 'w') as f:
            json.dump(config, f, indent=2)

    dataset_params = config['dataset_params']['pretrain']
    split_ratio = train_params.get('train_val_split', 0.9)
    # training_params.pretrain.seed (optional): weight init, masking and batch order all draw
    # from torch's RNG. The subject train/val split has its own fixed seed, so runs that
    # differ only in this seed share the same split. Unset = unseeded, as before.
    if 'seed' in train_params:
        torch.manual_seed(int(train_params['seed']))

    train_config = copy.deepcopy(config)
    val_config   = copy.deepcopy(config)

    # person-disjoint, order-independent subject split (IO/splits.py's split_pretrain_subjects)
    for ds_name, (train_subs, val_subs) in split_pretrain_subjects(dataset_params, split_ratio).items():
        train_config['dataset_params']['pretrain'][ds_name]['subject_to_use'] = train_subs
        val_config['dataset_params']['pretrain'][ds_name]['subject_to_use'] = val_subs
        logger.info(f"Dataset {ds_name}: {len(train_subs)} Train, {len(val_subs)} Val subjects")

    logger.info("Building Training Dataset...")
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='pretrain')
    logger.info("Building Validation Dataset...")
    val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='pretrain')
    logger.info(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    def _make_loader(dataset, shuffle):
        return DataLoader(dataset, batch_sampler=MontageBatchSampler(dataset, train_params['batch_size'], shuffle),
                          num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

    train_loader = _make_loader(train_dataset, shuffle=True)
    val_loader   = _make_loader(val_dataset,   shuffle=False)

    model_type = train_params.get('model_type', 'Qlet')
    entry      = MODEL_REGISTRY[model_type]
    trainer    = entry.trainer_cls()

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing model for {Nc} channels (Run: {model_name})...")
    model = build_pretrain_from_config(config, mode='pretrain')
    build_config = checkpoint_build_config(config, mode='pretrain')   # saved in every checkpoint

    tokenizer_epochs = train_params['tokenizer_epochs']
    freeze_atoms    = train_params.get('freeze_atoms', True)
    total_epochs     = train_params['epochs']
    model.enter_tokenizer_phase()
    model.to(device)
    logger.info(f"  [Tokenizer phase] epochs 1-{tokenizer_epochs}: every block, temporal only, unmasked. "
                f"Masked phase from epoch {tokenizer_epochs + 1}, "
                f"freeze_atoms={freeze_atoms}")

    logger.info("Warming up with dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, bool_masked_pos, valid_channels = unpack_batch(dummy_batch, device)
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx, bool_masked_pos=None, valid_channels=valid_channels)

    # Built once, while every param is trainable: bypassed blocks and (later) frozen
    # Qlets just get grad=None, which AdamW skips.
    scaler    = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(optimizer_param_groups(model, train_params['weight_decay']), lr=train_params['learning_rate'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    history = History()                       # stored as artifacts/history.json; the dashboard is drawn afterwards

    mask_pp     = config.get('preprocess_params', {}).get('mask', {})
    loss_params = config.get('model_params', {}).get(model_type, {}).get('pretrain', {}).get('loss', {})
    loss_hparams = dict(loss_params)
    # IO/masking.py: the strategy (with its optional channel subsampler) owns its schedule;
    # this loop only tells it the masked-phase epoch and redraws every masked epoch.
    mask_strategy = build_masking_strategy_from_config(mask_pp)
    logger.info(f"model_type={model_type}  masking={mask_pp.get('masking_strategy', 'random')}  "
                f"subsample={'on' if mask_strategy.subsampler else 'off'}  loss_hparams={loss_hparams}")

    best_val_loss = float('inf')  # reset at the phase boundary: masked loss isn't comparable
    logger.info(f"Starting {model_type} Pretraining ({total_epochs} epochs)")
    run_info = _run_info(config, model, train_dataset, val_dataset, device)
    run_info_path = os.path.join(artifact_dir, 'run_info.json')
    snapshots = {}                                         # per-epoch model state arrays -> artifacts/epoch_snapshots.npz
    checkpoint_every = train_params.get('checkpoint_every', 10)       # checkpoint/epoch_<n>.pth every n epochs (0 = never)

    def _write_run_info():
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
    _write_run_info()

    for epoch in range(1, total_epochs + 1):
        t_epoch, lr_epoch = time.time(), optimizer.param_groups[0]['lr']
        masked = epoch > tokenizer_epochs
        if epoch == tokenizer_epochs + 1:
            model.enter_masked_phase(freeze_atoms=freeze_atoms)
            best_val_loss = float('inf')
            logger.info(f"  [Masked phase] epoch {epoch}: all blocks, spatial attention + coord embedding on, "
                        f"AtomBank {'frozen' if freeze_atoms else 'TRAINING (mp loss stays on)'}")
        # Curriculum counts from the first masked epoch; masks are redrawn every masked epoch.
        # Tokenizer-phase epochs ignore the dataset's masks.
        mask_strategy.set_epoch(max(1, epoch - tokenizer_epochs))
        if masked:
            train_dataset.set_masking(mask_strategy)
            val_dataset.set_masking(mask_strategy)
            # rebuild, don't just re-iterate: persistent_workers=True means worker
            # subprocesses hold their own copy of the dataset from spawn time and never
            # see the mutation above otherwise (see PretrainDataset.set_masking).
            train_loader = _make_loader(train_dataset, shuffle=True)
            val_loader   = _make_loader(val_dataset,   shuffle=False)
            logger.info(f"  [mask curriculum] epoch {epoch}: {mask_strategy.describe()}")

        train_metrics = train_one_epoch(model, trainer, train_loader, optimizer, scaler, device, epoch, masked,
                                        **loss_hparams)
        val_metrics   = validate_one_epoch(model, trainer, val_loader, device, masked, **loss_hparams)
        scheduler.step()

        loss_keys = {'loss', 'masked', 'unmasked'}
        other_metrics = {k: v for k, v in train_metrics.items() if k not in loss_keys}

        logging.info(f"--- Epoch {epoch}/{total_epochs} ({'masked' if masked else 'tokenizer'}) Summary ---")
        logging.info(f"  [Train] " + " | ".join([f"{k}: {train_metrics.get(k, 0.0):.4f}" for k in ['loss', 'masked', 'unmasked']]))
        logging.info(f"  [Val]   " + " | ".join([f"{k}: {val_metrics.get(k, 0.0):.4f}"   for k in ['loss', 'masked', 'unmasked']]))
        # every loss term side by side, train against val (mse_band_* = relative error per band)
        loss_terms = [k for k in train_metrics if k.startswith('mse_')]
        if loss_terms:
            logging.info(f"  {'term':22s}{'train':>9s}{'val':>9s}")
            for k in loss_terms:
                logging.info(f"  {k:22s}{train_metrics[k]:9.4f}{val_metrics.get(k, float('nan')):9.4f}")
        other_metrics = {k: v for k, v in other_metrics.items() if not k.startswith('mse_')}

        if other_metrics:
            o_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in other_metrics.items()])
            logging.info(f"  [Other] {o_str}")

        logging.info("-" * 40)

        # Unconditional, every epoch: a mask-ratio curriculum (or any other future metric
        # surprise) can make val_metrics['loss'] structurally incomparable across epochs —
        # e.g. the mask-ratio ramp makes the task itself harder over
        # time, so best_val_loss below can freeze on an early, easy-ratio epoch and never
        # update again even while the model keeps genuinely improving within each step.
        # That leaves ONLY that early checkpoint on disk if training is later interrupted —
        # real instance: qlet_pretrain_v4 froze "best" at epoch 4/50, losing every epoch's
        # progress after that when the run was stopped at epoch 32. last.pth is the
        # insurance: whatever epoch you actually stopped at is always recoverable.
        torch.save({'model_state_dict': model.state_dict(), 'build_config': build_config}, os.path.join(checkpoint_dir, 'last.pth'))

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            # Tokenizer-phase best is overwritten by the first masked epoch (reset above).
            torch.save({'model_state_dict': model.state_dict(), 'build_config': build_config}, os.path.join(checkpoint_dir, 'best.pth'))
            logger.info(f"  > Saved Best Checkpoint ({'masked' if masked else 'tokenizer'} phase)")

        history.update(train_metrics=train_metrics, val_metrics=val_metrics,
                       epoch_info={'epoch': epoch, 'phase': 'masked' if masked else 'tokenizer', 'lr': lr_epoch,
                                   'seconds': time.time() - t_epoch, 'grad_scale': float(scaler.get_scale()),
                                   'mask': mask_strategy.describe() if masked else None})
        history.save(os.path.join(artifact_dir, 'history.json'))
        for k, v in trainer.epoch_snapshot(model).items():
            snapshots.setdefault(k, []).append(v)
        if snapshots:
            np.savez(os.path.join(artifact_dir, 'epoch_snapshots.npz'), epochs=np.arange(1, epoch + 1),
                     **{k: np.stack(v) for k, v in snapshots.items()})
        if checkpoint_every and epoch % checkpoint_every == 0:      # intermediate models: recon / atom figures at any epoch afterwards
            torch.save({'model_state_dict': model.state_dict(), 'build_config': build_config},
                       os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pth'))
        run_info.update(epochs_done=epoch, last_epoch_at=datetime.now().isoformat(timespec='seconds'))
        _write_run_info()

    run_info.update(status='complete', finished_at=datetime.now().isoformat(timespec='seconds'))
    _write_run_info()
    logger.info("Pretraining Complete.")


if __name__ == '__main__':
    main()
