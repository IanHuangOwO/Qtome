"""Closed-form probe pieces shared by the ridge-probe panels (ridge_probe, atom_vs_raw, coord_robustness): the Compass cells, the
alpha grid, the PCA of the latent z, and the Q-atom hidden features."""
import numpy as np
import torch
import torch.nn.functional as F


DATASETS = (('BNCI2014004', 'mi_loso'), ('BNCI2014001', 'mi_loso'), ('BNCI2014008', 'p300_loso'))


ALPHAS = np.logspace(-1, 5, 7)


def _pca_axes(tokens, n):
    """tokens [M, D] -> (mean [D], top-n principal axes [D, n]); eigh of the D x D covariance."""
    mu = tokens.mean(0)
    cov = (tokens - mu).T @ (tokens - mu) / max(len(tokens) - 1, 1)
    _, vecs = torch.linalg.eigh(cov.double())
    return mu, vecs[:, -n:].flip(1).to(tokens.dtype)


def _atom_bank(checkpoint):
    """The trained AtomBank of a pretrain checkpoint (rebuilt from its build_config)."""
    from model.factory import build_from_checkpoint
    return build_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False)).atoms.eval()


@torch.no_grad()
def atom_hidden(z, bank, chunk=256):
    """z [T, N', Cv, D] (the cached encoder output the Q-atoms read) -> Q-atom hidden u [T, N', Cv, S * K]:
    every Q-atom's MLP hidden GELU(LN(z) W_down_s + b_down_s), Q-atoms concatenated (AtomBank._amp's first map)."""
    out = []
    for c in z.split(chunk):
        u = F.gelu(torch.einsum('tncd,sdk->tncsk', bank.input_norm(c.float()), bank.W_down) + bank.b_down)
        out.append(u.flatten(-2))
    return torch.cat(out)


@torch.no_grad()
def token_rms(amp, u, bank):
    """Per-token input rms [T, N', Cv, 1], recovered from the cached amp = (u w_amp + b_amp) * rms by least
    squares over the Q-atoms (the cache keeps amp and z, not rms)."""
    out = []
    for a, h in zip(amp.split(256), u.split(256)):
        pre = torch.einsum('tncsk,skj->tncsj', h.unflatten(-1, (bank.W_down.shape[0], -1)), bank.w_amp) + bank.b_amp
        out.append((a.float() * pre).sum((-2, -1)) / pre.pow(2).sum((-2, -1)).clamp_min(1e-12))
    return torch.cat(out).unsqueeze(-1)


def _features(feature, data, checkpoint):
    """The token features a probe reads [T, N', Cv, F]: 'z' (as cached), 'atom_hidden' (u), 'ln_z' (z after the
    Q-atom bank's LayerNorm: z without its scale), 'atom_hidden_rms' (u times the token's input rms: scale put
    back) or 'ab' (the cached signed Q-atom gains a, b, rms included)."""
    z = data.z.float()
    if feature == 'z':
        return z
    if feature == 'ab':
        return data.amp.float().flatten(-2)
    bank = _atom_bank(checkpoint)
    if feature == 'ln_z':
        with torch.no_grad():
            return torch.cat([bank.input_norm(c) for c in z.split(256)])
    u = atom_hidden(z, bank)
    if feature == 'atom_hidden':
        return u
    assert feature == 'atom_hidden_rms', feature
    return u * token_rms(data.amp, u, bank)
