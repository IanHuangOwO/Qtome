"""The static qlet dictionary: every qlet active at every patch, each reconstructing a patch as a*D + b*H (D a template, H its quadrature partner)."""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomBank(nn.Module):
    """
    Static qlet dictionary over CHANNEL-GROUPED tokens: input is [G, C, D] where each group g is one
    patch POSITION (G = B*N) carrying all C channels' embeddings for that moment. Every qlet is active
    at every position; amplitude is read per channel. The instantaneous-mixing ICA picture made
    structural: x_c(t) = sum_s A[c, s] * source_s(t) -- D_hat_s is source s's waveform, and the [C]
    vector of a qlet's per-channel amps IS that source's mixing column (its topomap at that patch
    time). Routed (top-k) qlets were removed (`routed-stamps` branch).

    phi_s(z_c) = rms_c * (a_s(z_c) * D_hat_s + b_s(z_c) * Hilbert(D_hat_s)): a fixed per-atom waveform
    TEMPLATE D_s (no z dependence, used UNIT-L2-NORMALIZED everywhere -- with a free-norm D, amp*D has
    a scale degeneracy) plus its DERIVED Hilbert quadrature partner (see _quadrature), combined by a
    per-CHANNEL gain pair (a, b) from the qlet's own small MLP on z -- amplitude sqrt(a^2+b^2), phase
    atan2(b, a) -- times that channel's raw-input RMS (the LayerNorm stack erases amplitude from z, so
    the gain multiplies it back in). Shape is a pure parameter and amplitude/phase the only
    z-dependent knobs, so "same waveform, different amplitude across channels" is structural
    (no per-token shape warping). A free [patch_len] template's frequency content is
    bound to the patch_len FFT grid (Df = fs/patch_len); oscillator atoms were withdrawn (0010).
    """
    def __init__(self, dim, patch_len, n_atoms=16, hidden_width=16, hidden_layers=1):
        super().__init__()
        self.n_atoms = n_atoms
        self.hidden_layers = hidden_layers                                  # 1 = the original MLP z -> hidden -> (a, b); more = extra hidden -> hidden GELU maps
        # Normalizes z before it's used (z inherits whatever scale the encoder drifts to).
        self.input_norm = nn.LayerNorm(dim)
        # amp_s(z): per-atom MLP z -> hidden (GELU) -> quadrature gain pair (a, b). Because H is derived
        # from the SAME template, (a, b) can only re-phase and scale the shape, never morph it.
        self.W_down = nn.Parameter(torch.empty(n_atoms, dim, hidden_width))
        self.b_down = nn.Parameter(torch.zeros(n_atoms, hidden_width))
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        if hidden_layers > 1:                                               # created only then, so a one-hidden-layer checkpoint loads unchanged
            self.W_mid = nn.Parameter(torch.empty(hidden_layers - 1, n_atoms, hidden_width, hidden_width))
            self.b_mid = nn.Parameter(torch.zeros(hidden_layers - 1, n_atoms, hidden_width))
            for w in self.W_mid:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.w_amp = nn.Parameter(torch.empty(n_atoms, hidden_width, 2))
        self.b_amp = nn.Parameter(torch.zeros(n_atoms, 2))
        nn.init.kaiming_uniform_(self.w_amp, a=math.sqrt(5))
        # D_s: the qlet's waveform template, used unit-normalized; the raw parameter's norm is irrelevant.
        self.D = nn.Parameter(torch.randn(n_atoms, patch_len) * 0.02)

    def _amp(self, z):
        """z: [G, C, D] ALREADY input_norm'd tokens -> per-channel quadrature gains [G, C, n_atoms, 2],
        no rms. GELU between the two maps: without it they collapse into one of rank <= 2."""
        hidden = F.gelu(torch.einsum('gcd,hdk->gchk', z, self.W_down) + self.b_down)
        for l in range(self.hidden_layers - 1):
            hidden = F.gelu(torch.einsum('gchk,hkj->gchj', hidden, self.W_mid[l]) + self.b_mid[l])
        return torch.einsum('gchk,hkp->gchp', hidden, self.w_amp) + self.b_amp

    @staticmethod
    def _quadrature(D):
        """D: [M, L] unit templates -> each row's Hilbert quadrature partner [M, L], unit-normalized.
        Derived (rFFT, rotate every positive-frequency bin by -90 degrees, zero DC/Nyquist which have no
        quadrature, irFFT), NEVER a free parameter -- <D, H(D)> = 0 exactly, so a*D_hat + b*H_hat spans
        amplitude and constant phase of the template's analytic signal WITHOUT shape freedom.
        Re-normalized since zeroing DC/Nyquist drops that energy; an (almost-)pure-DC template's
        partner is degenerate -- its b head just learns ~0."""
        Fd = torch.fft.rfft(D.float(), dim=-1) * (-1j)
        Fd[..., 0] = 0
        if D.shape[-1] % 2 == 0:
            Fd[..., -1] = 0
        H = torch.fft.irfft(Fd, n=D.shape[-1], dim=-1)
        return F.normalize(H, dim=-1).to(D.dtype)

    def templates(self):
        """(D_hat, H_hat): unit templates and their quadrature partners, each [n_atoms, patch_len]."""
        D = F.normalize(self.D, dim=-1)
        return D, self._quadrature(D)

    def decode(self, amp):
        """amp: [G, C, n_atoms, 2] per-channel gains WITH rms (forward()'s out.amp) -> per-atom
        contribution [G, C, n_atoms, patch_len] = a*D_hat + b*H_hat, unsummed (forward's recon is its
        sum over qlets). Pure re-expansion, no model re-evaluation."""
        D, H = self.templates()
        return amp[..., 0, None] * D + amp[..., 1, None] * H

    @torch.no_grad()
    def dense_amp(self, z, rms=None):
        """z: [G, C, D] channel-grouped embeddings (same input forward() takes) -> amp [G, C, n_atoms, 2],
        input_norm'd and rms-scaled the same way forward() is."""
        amp = self._amp(self.input_norm(z))
        return amp if rms is None else amp * rms.unsqueeze(-1)

    def forward(self, z, rms=None, valid_channels=None, rank=False):
        """
        z: [G, C, D] channel-grouped token embeddings (G = B*N patch positions), rms: [G, C, 1] per-channel
        raw-input RMS or None -- multiplied into every amp; callers running the real pipeline always pass it.
        valid_channels: [G, C] bool (True = real channel) or None -- excludes zero-padded channels from h;
        padded channels still decode. rank: also return every qlet's decoded content in per-patch strength
        order (the nested reconstruction loss, QletPretrain._recon_loss).

        Returns recon [G, C, patch_len], amp [G, C, n_atoms, 2] (per-channel (a, b), rms included -- a
        qlet's [C] magnitude column is its phase-invariant topomap at this patch time), h [G, n_atoms]
        (post-rms amp magnitude averaged over valid channels: reconstruction-energy importance, the rank
        order) and ranked ([G, C, n_atoms, patch_len], strongest qlet first per patch; None unless rank).
        """
        G, C, _ = z.shape
        amp = self._amp(self.input_norm(z))                                 # [G, C, S, 2]
        if rms is not None:
            amp = amp * rms.unsqueeze(-1)                                   # restores raw amplitude

        energy = amp.pow(2).sum(dim=-1)                                     # [G, C, S]
        if valid_channels is not None:
            vc = valid_channels.unsqueeze(-1).to(energy.dtype)              # [G, C, 1]
            h = ((energy * vc).sum(dim=1) / vc.sum(dim=1).clamp(min=1.0)).clamp(min=0).sqrt()
        else:
            h = energy.mean(dim=1).clamp(min=0).sqrt()                      # [G, S]

        D, H = self.templates()
        recon = torch.einsum('gck,kl->gcl', amp[..., 0], D) + torch.einsum('gck,kl->gcl', amp[..., 1], H)

        # Ranked per patch by h, not by qlet index (a fixed global hierarchy): the nested loss's sets are each
        # patch's strongest k qlets (docs/adr/0018, adopted 2026-10-17; it replaced the MP residual loss).
        ranked = None
        if rank:
            contrib = self.decode(amp)                                      # [G, C, S, L]
            order = h.argsort(dim=-1, descending=True)                      # [G, S]
            ranked = contrib.gather(2, order[:, None, :, None].expand_as(contrib))

        return SimpleNamespace(recon=recon, amp=amp, h=h, ranked=ranked)
