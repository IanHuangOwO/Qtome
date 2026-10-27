"""ASR in the Q-atom space (panel task_filter_atom_asr): per Q-atom, the clean covariance over channels of its gains (a and b of every patch) gives
components and their clean rms; a component whose rms over a window's patches exceeds cutoff x its clean rms is set to zero for that atom and window."""
import torch


def _per_atom(g):                                                          # [B, N, C, S, 2] -> [S, B, 2N, C]: the [C] vectors of every patch's a and b, per Q-atom
    B, N, C, S, _ = g.shape
    return g.permute(3, 0, 1, 4, 2).reshape(S, B, N * 2, C).double()


def fit(g):
    """g: gains of the calibration windows [B, N, C, S, 2] -> (V [S, C, C] components, sd [S, C] their clean rms)."""
    x = _per_atom(g); S, B, M, C = x.shape
    cal = x.reshape(S, B * M, C); cov = cal.transpose(1, 2) @ cal / (B * M)
    ev, V = torch.linalg.eigh(cov)
    return V, ev.clamp_min(1e-20).sqrt()


def apply(g, model, cutoff):
    """-> (gains with the outlying components of every Q-atom zeroed, share of components removed per window [B])."""
    V, sd = model; B, N, C, S, _ = g.shape
    comp = _per_atom(g) @ V[:, None]                                              # [S, B, 2N, C]
    bad = comp.pow(2).mean(2) > (cutoff * sd[:, None, :]) ** 2             # [S, B, C]
    out = (comp * (~bad)[:, :, None, :]) @ V.transpose(1, 2)[:, None]
    return out.reshape(S, B, N, 2, C).permute(1, 2, 4, 0, 3).float(), bad.double().mean((0, 2)).numpy()


def atom_asr(g, calib, cutoff):
    """Calibrated on the windows `calib` (indices) of g itself."""
    return apply(g, fit(g[calib]), cutoff)
