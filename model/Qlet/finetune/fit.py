"""Fitting one head on a source: the SGD run (run_one) and the closed-form run (run_closed_form), with their metrics."""
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from IO.preprocessing import num_patches
from model.Qlet.finetune.sources import iter_batches
from model.Qlet.head.config import make_head_checkpoint, resolve_head_config
from model.Qlet.head.feature_head import FeatureHead
from model.history import History


def _metrics(labels, preds, loss):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {'loss': float(loss), 'acc': float((labels == preds).mean()),
                'f1': f1_score(labels, preds, average='macro', zero_division=0),
                'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
                'balanced_acc': balanced_accuracy_score(labels, preds), 'kappa': cohen_kappa_score(labels, preds)}


@torch.no_grad()
def _predict(head, source, idx, batch_size, device):
    """logits [n, C] (numpy) and mean loss over the trials idx, in the given (sorted) order."""
    head.eval()
    logits, ys = [], []
    for x, y in iter_batches(source, idx, batch_size, device, shuffle=False):
        logits.append(head(x).float()); ys.append(y)
    logits, ys = torch.cat(logits).cpu(), torch.cat(ys).cpu()
    return logits.numpy(), F.cross_entropy(logits, ys).item()


def build_head_factory(config, source, num_classes):
    """(resolved head config, function returning a freshly initialised FeatureHead)."""
    pp = config['preprocess_params']
    patch_len = pp.get('patch_length', 100)
    cfg = resolve_head_config(
        config['model_params']['Qlet']['finetune'], num_classes=num_classes, num_patches=source.num_patches,
        num_channels=len(source.channel_idx), num_atoms=source.num_atoms, patch_len=patch_len,
        latent_dim=getattr(source, 'latent_dim', None),
        patch_stride=pp.get('patch_stride', patch_len), sample_freq=float(pp['sample_freq']))
    return cfg, lambda: FeatureHead(cfg)


@torch.no_grad()
def _pca_axes(z, train_idx, m, chunk=256):
    """Top-m principal axes [m, D] of z [T, N', C, D] over the given trials (every patch and channel a row)."""
    idx = torch.as_tensor(train_idx, device=z.device)
    D = z.shape[-1]
    s, ss, n = z.new_zeros(D, dtype=torch.float64), z.new_zeros(D, D, dtype=torch.float64), 0
    for i in range(0, len(idx), chunk):
        x = z[idx[i:i + chunk]].reshape(-1, D).double()
        s += x.sum(0); ss += x.T @ x; n += len(x)
    mean = s / n
    cov = ss / n - torch.outer(mean, mean)
    return torch.linalg.eigh(cov)[1][:, -m:].flip(1).T.float()


def run_closed_form(config, run, source, tag, logger, out_dir=None):
    """training_params.finetune.fit 'closed_form': fit ClosedFormHead (model/Qlet/head/closed_form.py) on run['train'],
    no SGD; per-subject balanced accuracy + kappa on run['eval'], in run_one's format (tail = last: one fit)."""
    from model.Qlet.head.closed_form import ClosedFormHead
    tp = config['training_params']['finetune']
    if source.kind != 'atom':
        raise ValueError("fit 'closed_form' reads the atom code: the head needs a atom entry")
    amp, y = source.data.amp, source.labels.numpy()
    head = ClosedFormHead(**tp.get('closed_form', {})).fit(amp[run['train']].float().numpy(), y[run['train']])
    logger.info(f"[{tag}] closed_form {tp.get('closed_form', {})} train={len(run['train'])}")
    out, rows = {}, []
    for g, subs in run['eval'].items():
        sd = {}
        for s, i in subs.items():
            pred = head.predict(amp[i].float().numpy())
            rows.append((f'{g}/{s}', np.asarray(i), y[i], np.asarray(pred)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                acc, kappa = float(balanced_accuracy_score(y[i], pred)), float(cohen_kappa_score(y[i], pred))
            sd[s] = {'tail': acc, 'last': acc, 'kappa_tail': kappa, 'kappa_last': kappa, 'n_trials': int(len(i))}
        out[g] = {'subjects': sd, 'n_subjects': len(sd),
                  **{f'mean_{k}': float(np.mean([v[k] for v in sd.values()]))
                     for k in ('tail', 'last', 'kappa_tail', 'kappa_last')}}
    if out_dir:                                      # per-trial predictions: confusion matrices, ROC ... afterwards
        np.savez(os.path.join(out_dir, 'predictions.npz'), subjects=np.array([r[0] for r in rows]),
                 trial_index=np.concatenate([r[1] for r in rows]), y_true=np.concatenate([r[2] for r in rows]),
                 y_pred=np.concatenate([r[3] for r in rows]), counts=np.array([len(r[1]) for r in rows]))
    return out


def run_one(config, run, source, head_cfg, new_head, tag, out_dir, logger, device):
    """Train one head on run['train'], evaluate on the union of run['eval'] every epoch."""
    tp = config['training_params']['finetune']
    E, bs, seed, warm = tp['epochs'], tp['batch_size'], tp.get('seed', 42), tp['warmup_epochs']
    torch.manual_seed(seed)
    head = new_head().to(device)
    for mod in head.entries.values():            # latent_proj 'pca': fixed projection fit on this run's training trials
        if getattr(mod, 'e', {}).get('latent_proj') == 'pca':
            mod.proj.weight.data.copy_(_pca_axes(source.z, run['train'], mod.proj.out_features))
    opt = optim.AdamW(head.parameters(), lr=tp['learning_rate'], weight_decay=tp['weight_decay'])
    sched = optim.lr_scheduler.SequentialLR(
        opt, schedulers=[optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
                         optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, E - warm), eta_min=tp['min_learning_rate'])],
        milestones=[warm])
    ev_idx = np.unique(np.concatenate([i for g in run['eval'].values() for i in g.values()]))
    pos = {(g, s): np.searchsorted(ev_idx, i) for g, subs in run['eval'].items() for s, i in subs.items()}
    y_all = source.labels.numpy()
    y_ev = y_all[ev_idx]
    # training_params.finetune.class_weight "balanced": weight the loss by inverse class frequency
    # of this run's training trials (BNCI2014008 is 5:1 non-target, so plain CE predicts the majority)
    cw = None
    if tp.get('class_weight') == 'balanced':
        counts = np.bincount(y_all[run['train']], minlength=head_cfg['num_classes']).astype(float)
        cw = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    history = History()
    tail_start, hist, curves = max(0, E - 10), {}, {}
    logger.info(f"[{tag}] train={len(run['train'])} eval={len(ev_idx)} head_params={sum(p.numel() for p in head.parameters())}")
    for epoch in range(1, E + 1):
        t_epoch, lr_epoch = time.time(), opt.param_groups[0]['lr']
        head.train()
        losses, preds, ys = [], [], []
        for x, y in iter_batches(source, run['train'], bs, device, shuffle=True, gen=gen):
            opt.zero_grad()
            logits = head(x)
            loss = F.cross_entropy(logits, y, weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.detach()); preds.append(logits.argmax(1)); ys.append(y)   # no per-step GPU sync
        sched.step()
        train_metrics = _metrics(torch.cat(ys).cpu().numpy(), torch.cat(preds).cpu().numpy(),
                                 torch.stack(losses).mean().item())
        # eval batch size doesn't change predictions (eval mode: BatchNorm uses running stats)
        logits, val_loss = _predict(head, source, ev_idx, max(bs, 1024), device)
        val_pred = logits.argmax(1)
        val_metrics = _metrics(y_ev, val_pred, val_loss)
        with warnings.catch_warnings():              # per-subject balanced accuracy + kappa every epoch, from the same pass
            warnings.simplefilter('ignore')
            for key, p in pos.items():
                bk = (balanced_accuracy_score(y_ev[p], val_pred[p]), cohen_kappa_score(y_ev[p], val_pred[p]))
                curves.setdefault(key, []).append(bk)
                if epoch > tail_start:               # the tail (last 10 epochs) is what group_eval reports
                    hist.setdefault(key, []).append(bk)
        logger.info(f"--- [{tag}] Epoch {epoch}/{E} Summary ---")
        for name, m in (('Train', train_metrics), ('Val  ', val_metrics)):
            logger.info(f"  [{name}] loss: {m['loss']:.4f} | acc: {m['acc']:.4f} | f1: {m['f1']:.4f} | f1_w: {m['f1_weighted']:.4f}"
                        f" | bal_acc: {m['balanced_acc']:.4f} | kappa: {m['kappa']:.4f}")
        logger.info("-" * 40)
        history.update(train_metrics=train_metrics, val_metrics=val_metrics,
                       epoch_info={'epoch': epoch, 'lr': lr_epoch, 'seconds': time.time() - t_epoch})
    history.history['subjects'] = {f'{g}/{s}': {'bal_acc': [float(b) for b, _ in v], 'kappa': [float(k) for _, k in v]}
                                   for (g, s), v in curves.items()}
    history.save(os.path.join(out_dir['ckpt'], 'history.json'))
    logits_ev = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    np.savez(os.path.join(out_dir['ckpt'], 'predictions.npz'), trial_index=ev_idx, y_true=y_ev,        # last epoch's eval-set logits
             logits=logits_ev.astype(np.float32), subjects=np.array([f'{g}/{s}' for g, s in pos]),
             counts=np.array([len(p) for p in pos.values()]), positions=np.concatenate(list(pos.values())))
     # the dashboard is drawn afterwards (training_dashboard panel)
    torch.save(make_head_checkpoint(head, head_cfg, source.channel_idx, tp['pretrained_checkpoint']),
               os.path.join(out_dir['ckpt'], 'head.pth'))
    out = {}
    for g, subs in run['eval'].items():
        sd = {}
        for s, i in subs.items():
            bal_acc_hist, kappa_hist = zip(*hist[(g, s)])
            sd[s] = {'tail': float(np.mean(bal_acc_hist)), 'last': float(bal_acc_hist[-1]),
                      'kappa_tail': float(np.mean(kappa_hist)), 'kappa_last': float(kappa_hist[-1]),
                      'n_trials': int(len(i))}
        out[g] = {'subjects': sd, 'n_subjects': len(sd),
                  'mean_tail': float(np.mean([v['tail'] for v in sd.values()])),
                  'mean_last': float(np.mean([v['last'] for v in sd.values()])),
                  'mean_kappa_tail': float(np.mean([v['kappa_tail'] for v in sd.values()])),
                  'mean_kappa_last': float(np.mean([v['kappa_last'] for v in sd.values()]))}
    return out
