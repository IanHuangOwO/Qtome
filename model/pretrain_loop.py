"""One epoch of masked pretraining and one of validation, for any plugin model (its trainer supplies the loss)."""
import logging

import torch
from tqdm import tqdm

from IO.wrappers import MontageBatchSampler
from model.base_trainer import nonfinite_step_report


def unpack_batch(batch, device):
    x_patches, coords, mask, time_indices, _, valid_channels = batch
    x              = x_patches.to(device)      # [B, C, N, L]
    coords         = coords.to(device)         # [B, C, 3]
    time_idx       = time_indices.to(device)   # [B, N]
    valid_channels = valid_channels.to(device) # [B, C]
    B, C, N, L = x.shape
    # (C, N) order is implicit — mask itself carries no axis labels. Must match x_patches'
    # (C, N, L) layout in IO/dataset.py; any consumer reshaping (N, C) instead silently
    # misaligns masked/unmasked patches with no shape error.
    bool_masked_pos = mask.view(B, C, N).to(device)  # [B, C*N] -> [B, C, N]
    # Batches hold one montage (MontageBatchSampler): drop the canonical channels no window
    # in the batch has -- the model is channel-count agnostic, and they are pure padding.
    keep = valid_channels.any(0)
    if not keep.all():
        x, coords, bool_masked_pos, valid_channels = x[:, keep], coords[:, keep], bool_masked_pos[:, keep], valid_channels[:, keep]
    return x, coords, time_idx, bool_masked_pos, valid_channels


def _ratio_terms(metrics):
    """'<name>_num' / '<name>_den' pairs (epoch means of per-batch sums) -> '<name>' = their ratio, the ratio over
    the whole epoch rather than a mean of per-batch ratios (Qlet's per-band recon error)."""
    for k in [k for k in metrics if k.endswith('_num') and k[:-4] + '_den' in metrics]:
        num, den = metrics.pop(k), metrics.pop(k[:-4] + '_den')
        metrics[k[:-4]] = num / den if den > 0 else float('nan')
    return metrics


def train_one_epoch(model, trainer, data_loader, optimizer, scaler, device, epoch, masked,
                     **loss_hparams):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}
    skipped = 0

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, bool_masked_pos, valid_channels = unpack_batch(batch, device)
        # Tokenizer phase: the dataset still generates masks, they're just not used. Must be
        # None, not all-False — get_loss's None branch is the tokenizer phase's full reconstruction.
        if not masked:
            bool_masked_pos = None
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)
            l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, bool_masked_pos, **loss_hparams)
        # A non-finite loss must never reach backward (see nonfinite_step_report):
        # backwarding a non-finite loss can leave NaN parameters, which nothing recovers
        # from. update_diagnostics is skipped too — it folds `out` into EMA buffers.
        report = nonfinite_step_report(l_total, model, out)
        if report is not None:
            skipped += 1
            if skipped <= 3:
                logging.warning(f"epoch {epoch} batch {batch_idx} skipped: {report}")
            optimizer.zero_grad(set_to_none=True)
            continue

        trainer.update_diagnostics(model, out)

        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      # total norm before clipping
        scaler.step(optimizer)
        scaler.update()

        totals["loss"]     += l_total.item()
        totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
        totals["unmasked"] += l_unmasked.item()
        totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm)
        if hasattr(out, 'lb_loss'):
            totals["lb_loss"] = totals.get("lb_loss", 0.0) + (out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss))
        for name, v in (getattr(model, '_last_loss_terms', None) or {}).items():
            key = f'mse_{name}'
            totals[key] = totals.get(key, 0.0) + v

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            pbar.set_postfix({
                'L':   f"{totals['loss']     / n:.4f}",
                'msk': f"{totals['masked']   / n:.4f}",
                'vis': f"{totals['unmasked'] / n:.4f}",
            })

    n = max(batch_idx + 1 - skipped, 1)
    epoch_metrics = _ratio_terms({k: v / n for k, v in totals.items()})
    epoch_metrics.update(trainer.epoch_metrics(model, out))
    if skipped:
        logging.warning(f"epoch {epoch}: skipped {skipped}/{batch_idx + 1} non-finite batches")
        epoch_metrics['skipped_batches'] = skipped

    return epoch_metrics


def validate_one_epoch(model, trainer, data_loader, device, masked, **loss_hparams):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, bool_masked_pos, valid_channels = unpack_batch(batch, device)
            if not masked:
                bool_masked_pos = None

            with torch.amp.autocast(device_type='cuda'):
                out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)
                l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, bool_masked_pos, **loss_hparams)
            trainer.update_diagnostics(model, out)

            totals["loss"]     += l_total.item()
            totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
            totals["unmasked"] += l_unmasked.item()
            if hasattr(out, 'lb_loss'):
                totals["lb_loss"] = totals.get("lb_loss", 0.0) + (out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss))
            for name, v in (getattr(model, '_last_loss_terms', None) or {}).items():
                key = f'mse_{name}'
                totals[key] = totals.get(key, 0.0) + v

            if batch_idx % 5 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'L':   f"{totals['loss']     / n:.4f}",
                    'msk': f"{totals['masked']   / n:.4f}",
                    'vis': f"{totals['unmasked'] / n:.4f}",
                })

    n = batch_idx + 1
    val_metrics = _ratio_terms({k: v / n for k, v in totals.items()})
    val_metrics.update(trainer.epoch_metrics(model, out))

    return val_metrics
