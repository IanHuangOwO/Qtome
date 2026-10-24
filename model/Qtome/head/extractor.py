"""Q-atom codes (and optionally z) from the frozen backbone: what the finetune head reads."""
import torch
import torch.nn as nn


class AtomExtractor(nn.Module):
    """Everything that needs the frozen backbone: Q-atom code (a, b) per patch, channel and
    Q-atom, scaled by patch RMS. Output [B, N', C_valid, S, 2] float32, padded channels dropped."""
    def __init__(self, backbone, channel_idx):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('channel_idx', torch.as_tensor(channel_idx, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x, coords, time_idx=None, valid_channels=None, impute_missing=False, return_z=False):
        """return_z: False, True / 'output' (the encoder output the Q-atoms read) or 'bottleneck' (the
        deepest stage, linearly upsampled back to the patch grid)."""
        """impute_missing (experiment, 2026-09-25): feed every missing channel as the pretrain
        mask_token at its own coordinate (coords must hold real positions for them), so spatial
        attention fills it in, and scale its Q-atom code by the mean patch RMS of its 3 nearest
        real channels (its own RMS is 0). Every channel then counts as valid."""
        B, C, N, L = x.shape
        vmask = (valid_channels if valid_channels is not None
                 else x.new_ones(B, C, dtype=torch.bool))
        rms = x.float().pow(2).mean(-1).sqrt()                                            # [B, C, N]
        masked = None
        if impute_missing:
            miss = ~vmask
            masked = miss[:, :, None].expand(B, C, N)
            d = torch.cdist(coords.float(), coords.float())                                # [B, C, C]
            d = d.masked_fill(miss[:, None, :], float('inf'))                              # only real channels as neighbours
            nn_idx = d.topk(min(3, int(vmask.sum(1).min())), dim=-1, largest=False).indices  # [B, C, k]
            nn_rms = torch.gather(rms[:, None].expand(B, C, C, N), 2,
                                  nn_idx[..., None].expand(-1, -1, -1, N)).mean(2)          # [B, C, N]
            rms = torch.where(miss[:, :, None], nn_rms, rms)
            vmask = torch.ones_like(vmask)
        z, _ = self.backbone.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=masked,
                                            valid_channels=vmask)                       # [B, C, N, D]
        zg = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
        rg = rms.permute(0, 2, 1).reshape(B * N, C, 1)
        amp = self.backbone.atoms.dense_amp(zg, rms=rg)
        amp = amp.reshape(B, N, C, -1, 2).float() * vmask.float()[:, None, :, None, None]
        amp = amp[:, :, self.channel_idx]                                                 # [B, N, Cv, S, 2]
        if return_z:   # z for the latent_* head entries: [B, N, Cv, D]
            if return_z == 'bottleneck':
                z = self.backbone.encoder.upsample_bottleneck(N)
            return amp, z.permute(0, 2, 1, 3)[:, :, self.channel_idx].float()
        return amp
