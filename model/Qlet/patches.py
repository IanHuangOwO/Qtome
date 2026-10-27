"""Overlap-add and fold of patches back to a continuous signal (a trapezoid crossfade, not a plain reshape)."""
import torch
import torch.nn.functional as F


def overlap_add_patches(patches, stride):
    """patches: [..., N, L] patch_stride-spaced, patch_len-long patches on the last two
    dims (any number of leading batch dims) -> [..., T] the real continuous signal they
    were sliced from, T = (N-1)*stride + L.

    A plain reshape(..., N*L) only gives the real signal back when stride == L
    (non-overlapping patches) — patch_stride has been < patch_len (some overlap) since
    before this codebase's current patch_len/stride values, so that reshape silently
    stacks every patch's FULL length back to back regardless of real overlap: the
    shared region between consecutive patches gets duplicated (inflating the apparent
    duration) and, for a differentiable model output like recon (unlike raw, whose
    duplicate copies are byte-identical — the same real samples read twice), the two
    patches' independently-computed views of that same shared moment generally
    disagree, showing a real discontinuity at every patch boundary. See
    model/Qlet/Qlet_codebook_checker.py's event dynamics for the diagnostic-display use of
    this, and QletPretrain._recon_loss for the differentiable training-loss use.

    Linear-crossfade overlap-add instead: each patch gets a trapezoidal window — ramps
    0->1 over the incoming overlap it shares with the PREVIOUS patch, flat 1 over its
    own unique hop, ramps 1->0 over the outgoing overlap it shares with the NEXT one;
    no ramp on a side with no neighbor (the first/last patch). Every output sample is
    the weight-normalized sum of every patch covering it — a true weighted average, not
    dependent on the window being exactly constant-overlap-add — so this degrades to
    the old exact reshape behavior when stride == L (overlap == 0: every weight is 1).
    One F.fold call (a sum over every patch covering each sample) for both the weighted signal
    and the weights."""
    *lead, N, L = patches.shape
    assert stride <= L, f"overlap_add_patches: stride ({stride}) > patch_len ({L}) leaves gaps unfilled"
    overlap = L - stride
    w = torch.ones(N, L, device=patches.device, dtype=patches.dtype)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap + 2, device=patches.device, dtype=patches.dtype)[1:-1]
        w[1:, :overlap] = ramp
        w[:-1, -overlap:] = ramp.flip(0)
    return (fold_sum(patches * w, stride) / fold_sum(w, stride).clamp(min=1e-8)).reshape(*lead, -1)


def fold_sum(patches, stride):
    """[..., N, L] patches -> [..., T]: every sample the SUM of the patch samples covering it."""
    *lead, N, L = patches.shape
    T = (N - 1) * stride + L
    flat = patches.reshape(-1, N, L).transpose(1, 2)                        # [M, L, N]
    return F.fold(flat, (1, T), (1, L), stride=(1, stride)).reshape(*lead, T)
