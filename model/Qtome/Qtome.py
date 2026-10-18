import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Qtome.Qtome_modules import (SpatialTemporalEmbeddings, TSAEncoder, AtomBank, RelativeSpatialBias,
                                         fold_sum,
                                         spatial_mix, FlatTimePool, LearnedTimePool,
                                         AtomExtractor, FeatureHead,
                                         resolve_head_config, make_head_checkpoint,
                                         needs_atom, needs_raw, needs_latent, feature_names)


def _ema_update(buf, val, decay=0.99):
    """In-place EMA update, skipped if val is NaN/Inf. A plain `.mul_(decay).add_(val,
    alpha=1-decay)` permanently poisons buf the moment val is ever NaN even once (a single
    bad batch, e.g. a transient fp16-autocast overflow early in training): NaN propagates
    through every future update (decay*NaN + (1-decay)*anything = NaN), so a diagnostic
    stuck NaN for an entire run can trace back to one early outlier batch long since
    recovered from. Skipping non-finite updates lets the EMA keep tracking real values."""
    if torch.isfinite(val):
        buf.mul_(decay).add_(val, alpha=1 - decay)


def _restore_phase(module, incompatible_keys):
    """load_state_dict post-hook: re-apply the checkpoint's phase flags (plain
    attributes, not state). Does not freeze anything."""
    if bool(module.masked_phase):
        module.enter_masked_phase(freeze_atoms=False)
    else:
        module.enter_tokenizer_phase()


class QtomePretrain(nn.Module):
    """
    Spatiotemporal Q-atom-dictionary EEG tokenizer. Goal is explainable, per-patch embeddings (not a discrete vocabulary):
    channel-count invariant (cross-dataset unification still matters) but NOT
    length-invariant (each patch keeps its own embedding, for temporal localization of
    events within a trial).

    Pipeline: encoder -> AtomBank (dictionary of fixed per-atom waveform templates, each
    presented at a per-channel, per-atom amplitude/phase read off that atom's own
    bottleneck) -> reconstruction, summed directly in patch space (no separate decoder
    stage); the dictionary is static (every Q-atom active everywhere).

    Trains in two phases of one run (train_pretrain.py, CONTEXT.md:
    Tokenizer stage / Masked stage):
    - enter_tokenizer_phase(): every block runs with temporal mixing only (spatial
      attention and the coordinate embedding off), no masking (bool_masked_pos=None) —
      encoder + AtomBank train jointly on single-channel features, so the Q-atom dictionary
      isn't built from cross-channel-mixed input.
    - enter_masked_phase(freeze_atoms): every block runs, spatial attention + coord
      embedding on, bool_masked_pos set. AtomBank optionally frozen.
    The phase is a buffer, so any load_state_dict restores it (_restore_phase).
    """
    def __init__(
        self,
        embed_dim=100,
        enc_depth=12,
        mlp_ratio=4.0,
        patch_len=20,
        spatial_heads=10,
        dropout=0.0,
        num_channels=1,
        spatial_embedding=True,
        n_atoms=16,
        atom_hidden_width=16,
        n_routed_ffn_experts=4,
        n_shared_ffn_experts=1,
        ffn_top_k=2,
        patch_stride=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        # patch_stride: which samples a masked run truly hides (_position_weights, 50% overlap).
        # Falls back to patch_len (non-overlapping) matching PretrainDataset.
        self.patch_stride = patch_stride or patch_len
        self.head_dim = embed_dim
        self.num_channels = num_channels

        self.embed   = SpatialTemporalEmbeddings(patch_len, embed_dim, spatial=spatial_embedding)
        self.encoder = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                   dropout=dropout,
                                   n_routed_ffn_experts=n_routed_ffn_experts, n_shared_ffn_experts=n_shared_ffn_experts,
                                   ffn_top_k=ffn_top_k)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        # Masked tokens swap their CONTENT for mask_token before the time/coord embeddings
        # are added (MAE convention), so a masked token still knows where and when it is.
        # Zero-padded (missing) channels are left out of spatial attention as keys (TSABlock).
        # spatial_embedding: Fourier coordinate embedding + a directional relative-position bias in
        # every block's spatial attention (RelativeSpatialBias), on or off together (the ablation).
        self.spatial_bias = RelativeSpatialBias(enc_depth, spatial_heads) if spatial_embedding else None

        self.atoms = AtomBank(embed_dim, patch_len, n_atoms=n_atoms, hidden_width=atom_hidden_width)
        # convenience alias — viz/checker code reads it off the model directly
        self.n_atoms = self.atoms.n_atoms
        self.atoms_frozen = False
        self.register_buffer('masked_phase', torch.tensor(False))
        self.register_load_state_dict_post_hook(_restore_phase)

        # EMA health of the FFN MoE routers (MoEFFN/FFNRouter, one per TSABlock, averaged
        # across blocks by TSAEncoder.forward), see update_ffn_router_metrics below.
        self.register_buffer('ema_ffn_router_entropy',  torch.tensor(0.0))
        self.register_buffer('ema_ffn_router_load_std', torch.tensor(0.0))
        self.register_buffer('ema_ffn_gate_entropy',    torch.tensor(0.0))

    def enable_coord_embed(self):
        """Coordinate embedding ONLY — each channel's token learns WHERE it is, with no
        cross-channel content mixing (that is enable_spatial's MHA half, below). Kept as
        a standalone manual/experimental toggle (no longer auto-called anywhere in the
        training path: it measurably did nothing during the Tokenizer stage, no matter how the embedding's
        own architecture was fixed, because per-channel content already differentiates
        channels enough for that stage's loss without it).

        Was originally meant to let amp_i(z_c) become position-aware even while Q-atoms
        stay single-channel (AtomBank can't otherwise tell "alpha at Oz" from "alpha at
        Fz" when the raw content happens to coincide) — a real idea, just not one the
        Tokenizer stage's own reconstruction objective rewards learning. enable_spatial
        (below) now enables this alongside cross-channel attention in the Pretrain
        stage instead, where position could plausibly matter for attending across
        channels."""
        self.embed.enable_spatial()

    def enable_spatial(self):
        self.embed.enable_spatial()
        self.encoder.enable_spatial()

    def enable_temporal(self):
        self.encoder.enable_temporal()

    def enter_tokenizer_phase(self):
        self.masked_phase.fill_(False)
        self.enable_temporal()

    def enter_masked_phase(self, freeze_atoms=True):
        """Freeze before enable_spatial: a frozen dictionary never sees mixed z. With
        freeze_atoms=False it trains on mixed z, so per-atom amp is no longer a
        source topomap."""
        self.masked_phase.fill_(True)
        if freeze_atoms:
            self.freeze_atoms()
        self.enable_temporal()
        self.enable_spatial()

    def freeze_atoms(self):
        """
        End of Tokenizer stage: lock AtomBank so the Masked stage's frozen reconstruction
        target stops moving: two-stage/sequential rather than joint-warmup-then-freeze.
        """
        for p in self.atoms.parameters():
            p.requires_grad_(False)
        self.atoms_frozen = True

    @torch.no_grad()
    def update_ffn_router_metrics(self, ffn_router_entropy, ffn_router_load_std, ffn_gate_entropy):
        """EMA smoothing of the FFN MoE router health — see TSAEncoder.forward (Qtome_modules.py) for where these three
        already-averaged-across-blocks values come from. Called from
        QtomeTrainer.update_diagnostics with out.ffn_router_entropy/out.ffn_router_load_std/
        out.ffn_gate_entropy, same call site as update_head_metrics."""
        _ema_update(self.ema_ffn_router_load_std, ffn_router_load_std)
        _ema_update(self.ema_ffn_router_entropy, ffn_router_entropy)
        _ema_update(self.ema_ffn_gate_entropy, ffn_gate_entropy)

    def stage_features(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """Returns (z [B, C, N, D], ffn_lb_loss scalar) — ffn_lb_loss is the summed
        load-balance loss of every TSABlock's MoEFFN (see Qtome_modules.TSAEncoder)."""
        z = self.embed(x, coords=coords, bool_masked_pos=bool_masked_pos,
                       mask_token=self.mask_token)  # [B, C, N, D]
        bias = self.spatial_bias(coords) if self.spatial_bias is not None and coords is not None else None
        # a window's zero-padded tail patches (zero on every channel) are left out of temporal attention
        valid_patches = x.abs().amax(dim=(1, 3)) > 0                                    # [B, N]
        return self.encoder(z, valid_channels, bias, valid_patches)  # [B, C, N, D], ffn_lb_loss

    @torch.no_grad()
    def encode_atoms(self, x, coords, time_idx=None, valid_channels=None):
        """Unmasked Q-atom code for analysis/viz: AtomBank output (recon [G, C, L], amp
        [G, C, n_atoms, 2] with rms, h [G, n_atoms]; G = B*N positions, b*N + n), no ranked contents.
        Same per-channel rms as forward() -- without it amp lacks its raw-amplitude factor."""
        B, C, N, L = x.shape
        z, _ = self.stage_features(x, coords, time_idx=time_idx, valid_channels=valid_channels)
        z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
        rms = x.permute(0, 2, 1, 3).reshape(B * N, C, L).pow(2).mean(dim=-1, keepdim=True).sqrt()
        vc_g = None if valid_channels is None else valid_channels.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
        return self.atoms(z_g, rms=rms, valid_channels=vc_g)

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """
        x: [B, C, N, L], coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool — None during the Tokenizer stage (no masking);
        pass real masks only in the Masked stage, once temporal/spatial mixing are enabled
        and the Q-atoms are frozen (see enable_temporal/enable_spatial/freeze_atoms).
        valid_channels: [B, C] bool, True=real (not zero-padded) channel, or None. Passed into
        AtomBank (padded channels stay out of h) and carried through on the returned
        SimpleNamespace so get_loss/_recon_loss can exclude padded channels from the loss — a
        zero-padded channel's "reconstruction" is meaningless signal, not a real target. Padded
        channels still decode/reconstruct like any other.
        returns SimpleNamespace(recon [B,C,N,L], h [G, n_atoms] Q-atom strengths and amp
        [G, C, n_atoms, 2] (G = B*N patch positions, b*N + n; AtomBank.decode re-expands amp),
        ranked [B, C, N, n_atoms, L] (every Q-atom's content, strongest first per patch: the nested loss), ffn_lb_loss scalar (TSABlock MoEFFN routers, summed across blocks), FFN
        router health.
        """
        B, C, N, L = x.shape

        z, ffn_lb_loss = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos,
                                              valid_channels=valid_channels)  # [B, C, N, D]
        # Channel-grouped layout for AtomBank: [B, C, N, *] -> permute to [B, N, C, *]
        # then merge (B, N) — adjacent after the permute, so the merge is a safe reshape
        # (docs/agents/reshape-pitfalls.md; permute forces a copy, contiguity handled by
        # reshape itself). One group = one patch position with all its channels.
        G = B * N
        z_g = z.permute(0, 2, 1, 3).reshape(G, C, -1)          # [G, C, D]
        x_g = x.permute(0, 2, 1, 3).reshape(G, C, L)           # [G, C, L] — for the per-channel rms

        # Per-channel raw-input RMS — the amplitude signal the LayerNorm stack erased
        # from z (embed.norm -> per-block norm_out -> atoms.input_norm), multiplied back
        # into every amp inside AtomBank. Masked positions get 1.0: their true patch is
        # hidden from the encoder, so feeding its RMS would leak the target's amplitude
        # into masked reconstruction — the encoder must predict a masked patch's loudness
        # through z, same as it always did.
        rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()     # [G, C, 1]

        vc_g = None
        if valid_channels is not None:
            # [B, C] -> broadcast over N -> [G, C]: real channels only in h.
            vc_g = valid_channels.unsqueeze(1).expand(B, N, C).reshape(G, C)

        if bool_masked_pos is not None:
            mask_g = bool_masked_pos.permute(0, 2, 1).reshape(G, C, 1)
            # Masking is generated channel-agnostic (IO/masking.py never sees
            # valid_channels), so a padded channel's patch can land inside the mask —
            # gate the override on valid-AND-masked, not masked alone, or a padded
            # channel's rms gets force-set to 1.0 here, fabricating a nonzero amp/recon
            # for a channel that's always exactly 0.
            if vc_g is not None:
                mask_g = mask_g & vc_g.unsqueeze(-1)
            rms = torch.where(mask_g, torch.ones_like(rms), rms)

        out = self.atoms(z_g, rms=rms, valid_channels=vc_g, rank=True)

        recon = out.recon.reshape(B, N, C, L).permute(0, 2, 1, 3)  # back to [B, C, N, L]

        return SimpleNamespace(
            recon=recon,
            h=out.h,
            amp=out.amp,   # [G, C, n_atoms, 2] (G = B*N), for AtomBank.decode
            # [B, C, N, S, L] (G = B*N rows split into adjacent B, N): rank-ordered content, the nested loss
            ranked=out.ranked.reshape(B, N, C, *out.ranked.shape[2:]).permute(0, 2, 1, 3, 4),
            ffn_lb_loss=ffn_lb_loss,
            ffn_router_entropy=self.encoder.last_ffn_router_entropy,
            ffn_router_load_std=self.encoder.last_ffn_router_load_std,
            ffn_gate_entropy=self.encoder.last_ffn_gate_entropy,
            valid_channels=valid_channels,
        )

    def _position_weights(self, x, bool_masked_pos, valid_channels, unmasked_weight):
        """-> (valid [B, C, N, 1], w [B, C, N, L], hidden [B, C, N, L] or None). valid: 1 on real
        channels AND real patches. A patch that is exactly zero on every channel is time padding
        (a window's zero tail, see IO/preprocessing.py's window_continuous_signal) -- z-scored
        real EEG is never all-zero across a whole patch. hidden: a SAMPLE counts as masked only
        if every patch covering it is masked -- with 50% patch overlap, the outer half of a run's
        first/last masked patch is also inside a visible neighbour, so it is visible content,
        not a reconstruction target. w: valid, times (1 on hidden samples, unmasked_weight on
        visible ones) in the masked phase."""
        B, C, N, L = x.shape
        valid = x.new_ones(B, C, 1, 1) if valid_channels is None \
            else valid_channels.view(B, C, 1, 1).to(x.dtype)
        real_patch = (x.abs().amax(dim=(1, 3)) > 0).to(x.dtype).view(B, 1, N, 1)
        valid = valid * real_patch
        if bool_masked_pos is None:
            return valid, valid.expand(B, C, N, L), None
        stride = self.patch_stride
        m = bool_masked_pos.to(x.dtype).unsqueeze(-1).expand(B, C, N, L)
        n_cover = fold_sum(torch.ones(N, L, device=x.device, dtype=x.dtype), stride)       # [T]
        hidden = (fold_sum(m, stride) > n_cover - 0.5).to(x.dtype).unfold(-1, L, stride)  # [B, C, N, L]
        return valid, valid * (hidden + unmasked_weight * (1.0 - hidden)), hidden

    BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45), 'high': (45, 90)}

    @classmethod
    def _band_errors(cls, e, x, valid, fs=200.0):
        """Relative reconstruction error per band for the log: error power / target power in each band (rFFT of every
        valid patch; 0 = perfect, 1 = predicting zero), all valid positions as mse_patch. Patch bins are 2 Hz wide at
        100 samples, so delta is the 2 Hz bin alone. shortcut: fs fixed at 200 Hz, as _bin_weights.
        Returned as per-band sums ('<band>_num' error power, '<band>_den' target power): train_pretrain.py divides the
        epoch sums, so the logged value is the ratio over the epoch. A per-batch ratio explodes on a batch with almost
        no power in a band (a single pure-50 Hz BI2014a window in val, 2026-10-17)."""
        L = e.shape[-1]; f = torch.fft.rfftfreq(L, 1 / fs, device=e.device)
        v = valid.expand(*e.shape[:3], 1)                                   # [B, C, N, 1]
        Pe, Px = (torch.fft.rfft(t, dim=-1).abs().pow(2) * v for t in (e, x))
        out = {}
        for b, (lo, hi) in cls.BANDS.items():
            sel = (f >= lo) & (f < hi)
            out[f'{b}_num'], out[f'{b}_den'] = float(Pe[..., sel].sum()), float(Px[..., sel].sum())
        return out

    @staticmethod
    def _bin_weights(x, wp, max_hz, alpha=1.0, fs=200.0):
        """Per-rFFT-bin weights of the whitened loss (docs/cards/2026-10-17-whitened-mse.md and
        2026-10-17-partial-whitening-mse-mp.md): x [B, C, N, L] target, wp [B, C, N] patch weights -> [L//2+1] =
        Parseval multiplicity x S^-alpha, S the weighted mean patch spectrum (detached, floored at 1 % of its mean;
        bins above max_hz weighted as the last bin below), normalised so an error with the batch's own spectrum scores
        as plain MSE. (|rfft(e)|^2 * weights).sum(-1) / L^2 is then a patch's whitened mean squared error. alpha 1 =
        full whitening, 0 = plain MSE. shortcut: fs fixed at the project's 200 Hz, pass it in if that changes."""
        L = x.shape[-1]; k = torch.arange(L // 2 + 1, device=x.device)
        mult = torch.full((len(k),), 2.0, device=x.device, dtype=x.dtype); mult[0] = 1
        if L % 2 == 0:
            mult[-1] = 1                                                    # Parseval: mean(e^2) = sum(mult |E|^2) / L^2
        with torch.no_grad():
            S = (torch.fft.rfft(x, dim=-1).abs().pow(2) * wp[..., None]).sum((0, 1, 2)) / wp.sum().clamp(min=1e-12)
            S = S.clamp(min=0.01 * S.mean())
            top = int((k * fs / L <= max_hz).nonzero().max())
            S = torch.cat([S[:top + 1], S[top].expand(len(S) - top - 1)])
            wk = 1.0 / S if alpha == 1.0 else S.pow(-alpha)
            wk = wk * (mult * S).sum() / (mult * S * wk).sum()
        return mult * wk

    # Nested sets (default): the strongest S/8, S/4, S/2 and all S Q-atoms, weighted 1/8, 1/8, 1/4, 1/2 -- half the
    # weight on the full reconstruction, so the first Q-atoms cannot dominate.
    NESTED_FRACTIONS, NESTED_WEIGHTS = (1 / 8, 1 / 4, 1 / 2, 1), (0.125, 0.125, 0.25, 0.5)

    def _recon_loss(self, recon, x, ranked, bool_masked_pos, valid_channels=None,
                    mse_patch_weight=1.0, unmasked_weight=1.0, nested_sizes=None, nested_weights=None,
                    whiten_max_hz=0.0, whiten_alpha=1.0):
        """Recon loss: the nested reconstruction loss (docs/adr/0018; adopted 2026-10-17 over 3 seeds,
        docs/cards/2026-10-17-nested-loss-retest.md) -- per patch, sum_k w_k * |x - (strongest s_k Q-atoms)|^2,
        Q-atoms ranked per patch by strength (ranked [B, C, N, S, L], AtomBank.forward), nothing detached. The last
        size is every Q-atom, so that term is the plain patch MSE (nested_sizes [S] + nested_weights [1] is plain MSE,
        bit-identical). The smaller sets penalise a Q-atom that repeats a stronger one's content: the anti-duplicate
        job of the MP residual loss it replaced (ADR 0011; MP code removed 2026-10-17). Defaults: NESTED_FRACTIONS /
        NESTED_WEIGHTS ([3, 6, 12, 24] at 24 Q-atoms).

        Position weights: padded channels 0; in the masked phase masked positions 1 and visible
        positions `unmasked_weight` (0 = standard MAE, loss on masked patches only; 1 = every
        position counts equally). With bool_masked_pos=None (tokenizer phase) every valid position
        is a target and unmasked_weight is unused.

        whiten_max_hz > 0: every nested term's error is spectrally whitened per patch (_bin_weights; whiten_alpha
        sets the strength), patches weighted by the mean of their samples' position weights, since whitening mixes a
        patch's samples. 0 = plain MSE.

        Removed terms, with the evidence: the overlap-added trial MSE (it let
        neighbouring patches disagree on their shared samples as long as the crossfade averaged
        out; without it seam disagreement fell 40%), a masked STFT loss (0019); a window level (0011).

        Logged mse_patch stays the plain all-valid-position MSE, comparable across phases
        whatever the weights, and mse_nested_<k> each set's plain error. masked/unmasked are a diagnostic split.
        Returns (weighted total, l_masked, l_unmasked).
        """
        recon, x = recon.float(), x.float()
        valid, w, hidden = self._position_weights(x, bool_masked_pos, valid_channels, unmasked_weight)

        def wmean(err, wt):
            wt = wt.expand_as(err)
            s = wt.sum()
            return (wt * err).sum() / s if s > 0 else err.new_zeros(())

        K = ranked.shape[3]
        nested_sizes = list(nested_sizes or [int(K * f) for f in self.NESTED_FRACTIONS])
        nested_weights = list(nested_weights or self.NESTED_WEIGHTS)
        assert nested_sizes == sorted(set(nested_sizes)) and nested_sizes[0] > 0 and nested_sizes[-1] == K, \
            f"nested_sizes {nested_sizes}: increasing, last = every Q-atom ({K})"
        assert len(nested_weights) == len(nested_sizes) and abs(sum(nested_weights) - 1) < 1e-6, \
            f"nested_weights {nested_weights}: one per size, summing to 1"
        sel = torch.as_tensor([k - 1 for k in nested_sizes], device=x.device)
        prefix = ranked.float().cumsum(dim=3).index_select(3, sel)                     # [B, C, N, P, L]
        nw = torch.as_tensor(nested_weights, dtype=x.dtype, device=x.device).view(1, 1, 1, -1, 1)
        if whiten_max_hz:
            wp = w.mean(-1)                                                            # [B, C, N]
            bw = self._bin_weights(x, wp, whiten_max_hz, whiten_alpha)
            err = (torch.fft.rfft(prefix - x.unsqueeze(3), dim=-1).abs().pow(2) * bw).sum(-1) / x.shape[-1] ** 2
            err = (err * nw[..., 0]).sum(dim=3)                                        # [B, C, N]
            s = wp.sum()
            total = mse_patch_weight * ((wp * err).sum() / s if s > 0 else err.new_zeros(()))
            with torch.no_grad():
                nested_err = (prefix - x.unsqueeze(3)).pow(2)
        else:
            nested_err = (prefix - x.unsqueeze(3)).pow(2)                              # [B, C, N, P, L]
            total = mse_patch_weight * wmean((nested_err * nw).sum(dim=3), w)

        patch_err = (recon - x).pow(2)
        with torch.no_grad():
            plain_patch = wmean(patch_err, valid)
            l_masked, l_unmasked = 1.0, plain_patch
            if bool_masked_pos is not None:
                l_masked = wmean(patch_err, valid * hidden)
                l_unmasked = wmean(patch_err, valid * (1.0 - hidden))
            bands = self._band_errors(recon - x, x, valid)
            nested_plain = {f'nested_{k}': wmean(nested_err[:, :, :, i], valid).item() for i, k in enumerate(nested_sizes)}
        # Logged as mse_<name> (train_pretrain.py reads _last_loss_terms; plugin.py matches 'mse_').
        self._last_loss_terms = {'patch': plain_patch.item(), **{f'band_{b}': v for b, v in bands.items()}, **nested_plain}
        return total, l_masked, l_unmasked

    def get_loss(self, x, recon, ranked, bool_masked_pos=None,
                 mse_patch_weight=1.0, unmasked_weight=1.0,
                 ffn_lb_loss=None, ffn_lb_weight=0.01, valid_channels=None,
                 nested_sizes=None, nested_weights=None, whiten_max_hz=0.0, whiten_alpha=1.0):
        """
        Returns (total, l_masked, l_unmasked).

        Reconstruction term is _recon_loss's nested loss (see its docstring for mse_patch_weight /
        unmasked_weight / nested_* / whiten_*). Dictionary shaping (no near-duplicate Q-atoms) is the nested
        sets' job; earlier attempts -- activation decorrelation / negentropy, then the MP residual loss
        (ADR 0011) -- were measured and dropped.

        Tokenizer stage (bool_masked_pos=None): full reconstruction, l_masked=1.0
        placeholder (nothing masked yet).

        ffn_lb_loss (MoEFFN routers' load-balance loss, summed across TSABlocks) is added unconditionally, both stages: it comes
        from the encoder, which keeps training through the Masked stage (freeze_atoms()
        never locks the encoder).
        """
        total, l_masked, l_unmasked = self._recon_loss(
            recon, x, ranked, bool_masked_pos, valid_channels=valid_channels,
            mse_patch_weight=mse_patch_weight, unmasked_weight=unmasked_weight,
            nested_sizes=nested_sizes, nested_weights=nested_weights,
            whiten_max_hz=whiten_max_hz, whiten_alpha=whiten_alpha)

        if ffn_lb_loss is not None:
            total = total + ffn_lb_weight * ffn_lb_loss
        return total, l_masked, l_unmasked

    def get_metrics(self):
        metrics = {}

        # Per-block contribution norm — direct measure of how much each encoder block
        # actually changes its input (not the skip-gate proxy above, which conflates
        # "shallow skip re-injected" with "deep processing did nothing"). For a block
        # right before a pool point, this already has that block's own gate folded in
        # (see TSAEncoder.forward) — its real surviving contribution, not just its raw
        # pre-gate delta. Only populated after an eval-mode forward pass
        # (validate_one_epoch), same convention as the other diagnostics that gate on
        # `not self.training`.
        block_norms = getattr(self.encoder, 'last_block_norms', None)
        if block_norms:
            for i, v in enumerate(block_norms):
                metrics[f'block_norm_{i}'] = v

        # Largest branch output magnitude anywhere in the encoder, measured BEFORE
        # LayerScale shrinks it (see TSABlock._watch). The norms bound each branch's input
        # and each block's output; nothing bounds the middle, and LayerScale hides it from
        # block_norm. Early warning: this climbs for epochs before a float ceiling is
        # actually crossed, while every other diagnostic still looks healthy — v13 died
        # that way. Judge it against the training dtype's ceiling (65504 for fp16); order
        # 1-10 is normal, hundreds means the next run dies whatever the loss curve says.
        # No separate headroom fraction: it is this number over a constant, and it rounds
        # to 0.0000 in the log across the entire healthy range.
        branch_max = getattr(self.encoder, 'last_branch_max', None)
        if branch_max is not None:
            metrics['branch_max'] = branch_max.item()

        # FFN router health — see update_ffn_router_metrics above (per-TSABlock MoEFFN routers,
        # averaged across blocks).
        metrics['ffn_router_entropy']  = self.ema_ffn_router_entropy.item()
        metrics['ffn_router_load_std'] = self.ema_ffn_router_load_std.item()
        metrics['ffn_gate_entropy']    = self.ema_ffn_gate_entropy.item()

        return metrics


class FinetuneModel(nn.Module):
    """Frozen Qtome backbone + one FeatureHead (Qtome_modules finetune section). Call signature
    matches the old finetune classes so train_finetune.py is unchanged: forward -> (logits, None, None)."""
    def __init__(self, backbone, head_cfg, channel_idx):
        super().__init__()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('channel_idx', torch.as_tensor(channel_idx, dtype=torch.long))
        self.head_cfg = head_cfg
        if needs_latent(head_cfg):
            raise NotImplementedError("latent_* head entries run from the feature cache (train_finetune.py) only")
        atom = needs_atom(head_cfg)
        self.extractor = AtomExtractor(backbone, channel_idx) if atom else None
        if atom:
            assert head_cfg['num_atoms'] == backbone.n_atoms, "num_atoms must equal the atom count"
        self.head = FeatureHead(head_cfg)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # frozen: no dropout noise
        return self

    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        B, C = x.shape[:2]
        vm = valid_channels if valid_channels is not None else x.new_ones(B, C, dtype=torch.bool)
        inp = {}
        with torch.no_grad():
            if self.extractor is not None:
                inp['atom'] = self.extractor(x, coords, time_idx, vm)
            if needs_raw(self.head_cfg):
                inp['raw'] = x[:, self.channel_idx] * vm[:, self.channel_idx].float()[:, :, None, None]
        return self.head(inp), None, None

    def head_checkpoint(self, backbone_checkpoint):
        return make_head_checkpoint(self.head, self.head_cfg, self.channel_idx.tolist(), backbone_checkpoint)

    @classmethod
    def from_checkpoint(cls, backbone, ckpt):
        cfg = dict(ckpt['head_config'])
        channel_idx = cfg.pop('channel_idx')
        model = cls(backbone, cfg, channel_idx)
        model.head.load_state_dict(ckpt['model_state_dict'])
        return model


def build_finetune(backbone, num_channels, num_classes, channel_idx=None, num_patches=None,
                   sample_freq=200, **ft_params):
    """finetune_cls entry: builds the frozen backbone + FeatureHead from the numeric head config."""
    channel_idx = list(range(num_channels)) if channel_idx is None else list(channel_idx)
    num_atoms = 0
    if needs_atom(ft_params):
        num_atoms = backbone.atoms.n_atoms
    cfg = resolve_head_config(ft_params, num_classes=num_classes, num_patches=num_patches,
                              num_channels=len(channel_idx), num_atoms=num_atoms,
                              patch_len=backbone.patch_len, patch_stride=backbone.patch_stride,
                              sample_freq=float(sample_freq))
    return FinetuneModel(backbone, cfg, channel_idx)
