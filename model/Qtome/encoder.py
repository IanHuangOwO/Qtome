"""The temporal -> spatial -> FFN encoder: TSABlock (LayerScale, FoPE temporal attention, spatial attention with the relative bias) and TSAEncoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Qtome.embeddings import FoPE, _rotate_half
from model.Qtome.ffn import MoEFFN


class TSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.0,
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2):
        super().__init__()
        self.norm_time = nn.LayerNorm(dim)
        # Same nn.MultiheadAttention as spatial_attn below, over the PATCH axis N
        # instead of the channel axis C — so temporal mixing goes through PyTorch's
        # fused SDPA/flash kernels too. Softmax attention is a convex combination of v,
        # so its output is bounded by v's own range regardless of block depth or scale
        # drift. Time position comes from FoPE on q and k (TSAEncoder passes the rotation).
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_t = nn.Dropout(dropout)
        # Bounds this branch's raw output BEFORE scale_t multiplies it (applied to
        # attn_out_t/attn_out/ffn_out in forward(), right after _watch measures the
        # UNbounded version) — see the LayerScale comment below for why this is needed on
        # top of LayerScale: without it, a branch's own weights can grow arbitrarily to
        # counteract a tiny scale_t/s/ffn init, which is exactly what branch_max measures.
        self.norm_time_out = nn.LayerNorm(dim)

        self.norm_space = nn.LayerNorm(dim)
        # dropout here is on the attention WEIGHTS themselves (nn.MultiheadAttention's own
        # `dropout` arg), on top of drop_s below which drops the branch's output — two
        # different regularization points, same shared `dropout` value.
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_s = nn.Dropout(dropout)
        self.norm_space_out = nn.LayerNorm(dim)  # see norm_time_out

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MoEFFN(dim, hidden_dim=int(dim * mlp_ratio), n_routed=n_routed_ffn_experts,
                           n_shared=n_shared_ffn_experts, top_k=ffn_top_k,
                           dropout=dropout)
        self.norm_ffn_out = nn.LayerNorm(dim)  # see norm_time_out

        # norm_time/norm_space/norm_ffn are pre-norm (normalize the input to each
        # sub-layer) — nothing caps the residual stream x itself after three unbounded
        # adds, so later blocks operating on an already-inflated x can blow up further
        # (observed: block_norm_10/11 growing ~13x over 17 Pretrain epochs while a frozen
        # SAE dictionary downstream can't adapt to the drifting scale, see dead_feature_rate
        # climb). This final norm caps x's own magnitude every block.
        self.norm_out = nn.LayerNorm(dim)

        # LayerScale (Touvron et al., CaiT) on all three residual branches: a learnable
        # per-channel multiplier, init small, applied to each branch's output before the
        # residual add. Unlike the zero-init below (a one-time starting condition on two of
        # the three branches only, nothing stopping unbounded growth afterward), this stays
        # active for the whole run and throttles each branch's net CONTRIBUTION to x. It
        # does NOT bound the branch's own INTERNAL magnitude, though — LayerScale only
        # multiplies the branch's output on its way out, so gradient descent can (and does:
        # measured branch_max ~2500 on a converged FFN branch after 50 epochs, out_proj/fc2
        # weights growing to counteract scale_ffn's tiny init rather than the branch
        # learning to stay small) inflate the branch's own weights to reach whatever
        # effective magnitude it wants regardless of how small scale_t/s/ffn is. norm_*_out
        # (added right after each branch, before its scale multiply — see norm_time_out)
        # closes that loop: once the branch's raw output is itself normalized, scale_t/s/ffn
        # alone controls the residual contribution, removing the incentive to grow the
        # branch's internal weights in the first place. LayerScale still matters on top of
        # that (controls how much of the now-bounded branch reaches the residual stream).
        layerscale_init = 1e-4
        self.scale_t   = nn.Parameter(torch.full((dim,), layerscale_init))
        self.scale_s   = nn.Parameter(torch.full((dim,), layerscale_init))
        self.scale_ffn = nn.Parameter(torch.full((dim,), layerscale_init))

        # Both cross-patch (temporal, global context pooled over all N) and cross-channel
        # (spatial) mixing default off — Qtome's tokenizer stage trains the SAE on
        # patch-local features only, so the frozen dictionary can't leak already-seen
        # context into masked-stage reconstruction targets. Both out_proj-equivalents are
        # zero-inited so enabling later starts as a no-op and grows in under gradient,
        # instead of shocking a checkpoint that never saw either term active.
        self.temporal_active = False
        self.spatial_active = False
        self.last_branch_max = None  # see _watch
        nn.init.zeros_(self.temporal_attn.out_proj.weight)
        nn.init.zeros_(self.temporal_attn.out_proj.bias)
        nn.init.zeros_(self.spatial_attn.out_proj.weight)
        nn.init.zeros_(self.spatial_attn.out_proj.bias)

    def enable_temporal(self):
        self.temporal_active = True

    def enable_spatial(self):
        self.spatial_active = True

    def _watch(self, t):
        """Track the largest magnitude any branch produces BEFORE norm_*_out/LayerScale
        shrink it — called on the raw attn_out_t/attn_out/ffn_out, ahead of both.

        This is the blind spot that cost a tokenizer run. norm_time/norm_space/norm_ffn
        bound each branch's INPUT and norm_out bounds the block's OUTPUT, but nothing
        bounded what happened between them — and scale_t/scale_s/scale_ffn (init 1e-4)
        multiply the branch output on its way to the residual add, so an enormous
        interior arrived at the stream as a whisper. block_norm measures the post-scale
        contribution, i.e. the wrong side of that multiplication: it can stay tame right
        up to the moment a branch's interior crosses fp16's 65504 ceiling. norm_*_out
        (see the LayerScale comment above) now closes that gap architecturally, but this
        stays a raw pre-norm probe on purpose — it's the canary for the underlying matmul
        itself, independent of whatever norm_*_out does downstream.

        Deliberately NOT gated on `not self.training`, unlike last_block_norms: the
        overflow happened in TRAIN mode (eval on the same weights was finite), so an
        eval-only probe cannot see this class of failure at all. Cost is one amax
        reduction per branch — negligible against the block's matmuls — and it stays a
        tensor here so no per-step GPU sync happens; the single .item() is paid once per
        epoch in get_metrics.
        """
        m = t.detach().abs().amax()
        self.last_branch_max = m if self.last_branch_max is None else torch.maximum(self.last_branch_max, m)

    def _spatial_attention(self, x, valid_channels, spatial_bias):
        """x [B, N, C, D] -> [B, N, C, D]. spatial_attn's own weights, run through
        scaled_dot_product_attention so the per-block bias [B, H, C, C] and the padded-channel
        key mask broadcast over the N patches instead of being copied B*N times."""
        B, N, C, D = x.shape
        mha = self.spatial_attn
        H = mha.num_heads
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, C, H, D // H).transpose(2, 3) for t in (q, k, v))   # [B, N, H, C, d]
        mask = None
        if spatial_bias is not None:
            mask = spatial_bias.to(q.dtype)[:, None]                                    # [B, 1, H, C, C]
        if valid_channels is not None:
            pad = torch.zeros(B, 1, 1, 1, C, dtype=q.dtype, device=q.device).masked_fill(
                ~valid_channels.bool()[:, None, None, None, :], float('-inf'))
            mask = pad if mask is None else mask + pad
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=mha.dropout if self.training else 0.0)
        return mha.out_proj(out.transpose(2, 3).reshape(B, N, C, D))

    def _temporal_attention(self, x, kpm, temporal_rot):
        """x [B*C, N, D] -> [B*C, N, D]. Without a bias: temporal_attn as a plain MHA call. With
        one ([H, N, N]): the same weights through scaled_dot_product_attention, the bias and the
        padded-patch key mask broadcast over B*C."""
        mha = self.temporal_attn
        BC, N, D = x.shape
        H = mha.num_heads
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(BC, N, H, D // H).transpose(1, 2) for t in (q, k, v))       # [BC, H, N, d]
        cos, sin = (t.to(q.dtype)[None] for t in temporal_rot)                            # FoPE, [1, H, N, d]
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        mask = None
        if kpm is not None:
            mask = torch.zeros(BC, 1, 1, N, dtype=q.dtype, device=q.device).masked_fill(
                kpm[:, None, None, :], float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=mha.dropout if self.training else 0.0)
        return mha.out_proj(out.transpose(1, 2).reshape(BC, N, D))

    def forward(self, x, valid_channels=None, spatial_bias=None, valid_patches=None, temporal_rot=None):
        """valid_channels [B, C] bool (optional): zero-padded channels are left out of
        spatial attention as keys, so a montage's missing channels can't dilute the
        softmax (their own rows still get computed, then ignored downstream).
        valid_patches [B, N] bool (optional): the same for temporal attention over a
        window's zero-padded tail patches."""
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)
        self.last_branch_max = None

        if self.temporal_active:
            x_norm_t = self.norm_time(x_flat)
            kpm_t = None if valid_patches is None else \
                (~valid_patches.bool()).repeat_interleave(C, dim=0)             # [B*C, N], True = ignore
            attn_out_t = self._temporal_attention(x_norm_t, kpm_t, temporal_rot)
            self._watch(attn_out_t)
            attn_out_t = self.norm_time_out(attn_out_t)
            x_flat = x_flat + self.drop_t(self.scale_t * attn_out_t)

        x_space = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.spatial_active:
            x_norm = self.norm_space(x_space).view(B, N, C, D)
            attn_out = self._spatial_attention(x_norm, valid_channels, spatial_bias).reshape(B * N, C, D)
            self._watch(attn_out)
            attn_out = self.norm_space_out(attn_out)
            x_space = x_space + self.drop_s(self.scale_s * attn_out)
        x_flat = x_space.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)

        ffn_out, ffn_lb_loss = self.ffn(self.norm_ffn(x_flat))
        self._watch(ffn_out)
        ffn_out = self.norm_ffn_out(ffn_out)
        x_flat = x_flat + self.scale_ffn * ffn_out
        x_flat = self.norm_out(x_flat)
        return x_flat.view(B, C, N, D), ffn_lb_loss


class TSAEncoder(nn.Module):
    """depth TSABlocks at one resolution: every block sees all N patches of every channel. (Until
    2026-10-13 the patch axis was pooled by 2 between stages of blocks and upsampled back through gated
    skips; the flat stack reconstructs better at the same parameter count, docs/cards/2026-10-12-no-pooling.md.)
    Time position: FoPE on every block's temporal attention (fope_train_len = patches per training window)."""
    def __init__(self, dim, depth=8, num_heads=8, mlp_ratio=4., dropout=0.0,
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2,
                 fope_train_len=19):
        super().__init__()
        block = lambda: TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                                 n_routed_ffn_experts=n_routed_ffn_experts,
                                 n_shared_ffn_experts=n_shared_ffn_experts, ffn_top_k=ffn_top_k)
        self.blocks = nn.ModuleList([block() for _ in range(depth)])
        self.fope = FoPE(num_heads, dim // num_heads, fope_train_len)

    def enable_spatial(self):
        for block in self.blocks:
            block.enable_spatial()

    def enable_temporal(self):
        for block in self.blocks:
            block.enable_temporal()

    def forward(self, x, valid_channels=None, spatial_bias=None, valid_patches=None):
        """spatial_bias: optional [B, depth, H, C, C] from RelativeSpatialBias, block i gets [:, i].
        valid_patches: optional [B, N] bool, False on a window's zero-padded tail patches (left out
        of temporal attention as keys)."""
        # Per-block contribution norm (eval only): how much each block changes its input.
        record_norms = not self.training
        if record_norms:
            self.last_block_norms = []
        ffn_lb_loss = x.new_zeros(())
        rot = self.fope(torch.arange(x.shape[2], device=x.device, dtype=torch.float32))   # token times
        for i, block in enumerate(self.blocks):
            x_in = x
            x, blk_ffn_lb = block(x, valid_channels, None if spatial_bias is None else spatial_bias[:, i],
                                  valid_patches, rot)
            ffn_lb_loss = ffn_lb_loss + blk_ffn_lb
            if record_norms:
                with torch.no_grad():
                    self.last_block_norms.append((x - x_in).norm(dim=-1).mean().item())

        # Router health and the worst interior branch magnitude (TSABlock._watch), over every block.
        with torch.no_grad():
            blocks = self.blocks
            self.last_ffn_router_entropy = torch.stack([b.ffn.last_router_entropy for b in blocks]).mean()
            self.last_ffn_router_load_std = torch.stack([b.ffn.last_router_load_std for b in blocks]).mean()
            self.last_ffn_gate_entropy = torch.stack([b.ffn.last_gate_entropy for b in blocks]).mean()
            watched = [b.last_branch_max for b in blocks if b.last_branch_max is not None]
            self.last_branch_max = torch.stack(watched).max() if watched else None

        return x, ffn_lb_loss
