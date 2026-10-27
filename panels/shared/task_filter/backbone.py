"""The frozen backbone a task filter is built on and applied through: templates, dominant frequencies, decoding of gains, and the imputing encoder."""
import numpy as np
import torch

from IO.preprocessing import slice_patches
from model.factory import build_from_checkpoint
from model.Qtome.atom_bank import AtomBank
from model.Qtome.head.extractor import AtomExtractor
from model.Qtome.patches import overlap_add_patches
from panels.shared.task_filter.cells import CHANNELS, FS, layout, names, slot_map


class Backbone:
    """The frozen backbone of output/Qtome/<backbone>: the model, its Q-atom templates D and quadrature partners H,
    their dominant frequencies, and decoding of gains [B, N, C, S, 2] to waveforms [B, C, T]."""

    def __init__(self, bb):
        ck = torch.load(f'{bb}/pretrain/checkpoint/last.pth', map_location='cpu', weights_only=False)
        self.model = build_from_checkpoint(ck).eval()
        D = ck['model_state_dict']['atoms.D']; P = np.abs(np.fft.rfft((D - D.mean(1, keepdim=True)).numpy(), axis=1)) ** 2
        self.freqs = np.fft.rfftfreq(D.shape[1], 1 / FS)[P.argmax(1)]    # [S] peak of each template's spectrum
        D = torch.nn.functional.normalize(D, dim=-1); self.D, self.H = D.numpy(), AtomBank._quadrature(D).numpy()

    def patches(self, g):                                                 # gains [B, N, C, S, 2] -> patches [B, C, N, L]
        g = g.numpy() if torch.is_tensor(g) else g
        return np.einsum('tncs,sl->tcnl', g[..., 0], self.D) + np.einsum('tncs,sl->tcnl', g[..., 1], self.H)

    def decode(self, g):                                                  # -> [B, C, T], overlap-add at the 50% patch stride
        return overlap_add_patches(torch.from_numpy(self.patches(g)), self.D.shape[1] // 2).numpy()


def imputed_encoder(bb, ds, model, coords64):
    """encode(w [B, C, T] of ds's real channels in gain order, impute) -> gains in trial units [B, N, C or 64, S, 2]: of
    the real channels only, or (impute) of all 64 slots, the missing ones as mask tokens at their template coordinates."""
    valid, coords, pp = layout(bb, ds); lab, slot = slot_map(ds)
    idx = torch.tensor(sorted(slot[n] for n in lab if n in CHANNELS))      # the montage's own channels only: another channel sits in a
    valid = torch.zeros_like(valid); valid[idx] = True                      # free slot at its own position, which the filter does not know
    assert idx.tolist() == [slot[n] for n in names(ds)[1] if n in CHANNELS] and torch.allclose(coords[idx], coords64[idx], atol=1e-5)
    ext, ext64 = AtomExtractor(model, idx.tolist()).eval(), AtomExtractor(model, list(range(64))).eval()

    def encode(w, impute, want_z=False):                                   # want_z: also the encoder output z [B, N, C or 64, D] the Q-atoms read
        w = torch.from_numpy(w).float(); mu, s_ = w.mean((1, 2), keepdim=True), w.std((1, 2), keepdim=True)
        x = torch.zeros(len(w), len(valid), w.shape[-1]); x[:, idx] = (w - mu) / s_; out = []
        with torch.no_grad():
            for i in range(0, len(w), 64):
                xp, _ = slice_patches(x[i:i + 64], pp['patch_length'], pp['patch_stride']); b, P = xp.shape[0], xp.shape[2]
                out.append((ext64 if impute else ext)(xp, coords64[None].expand(b, -1, -1), torch.arange(P)[None].expand(b, P), valid[None].expand(b, -1), impute_missing=impute, return_z=want_z))
        if want_z:
            return torch.cat([o[0] for o in out]) * s_[:, :, :, None, None], torch.cat([o[1] for o in out])
        return torch.cat(out) * s_[:, :, :, None, None]
    return encode
