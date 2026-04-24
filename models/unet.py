"""Dual-encoder UNet for query-conditioned stem separation."""
import torch
import torch.nn as nn

from models.conv_blocks import ConvBlock, FiLM, TimestepEmbedding
from models.attention import BottleneckAttention


class EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, use_film: bool):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.film = FiLM(cond_dim, out_ch) if use_film else None
        self.pool = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled, skip)."""
        x = self.conv(x)
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        return self.pool(x), x


class DecoderLevel(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
        self.film = FiLM(cond_dim, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims if odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.film(x, cond)


class StemSeparationUNet(nn.Module):
    """
    Dual encoder UNet:
      - stem encoder: receives noisy stem latent, conditioned via FiLM(clap + t)
      - mix encoder:  receives mixture latent, no FiLM conditioning
    Bottleneck: BottleneckAttention (stem self-attention + cross-attention to mixture).
    Decoder: FiLM-conditioned, skips from stem encoder.
    Output: predicted noise matching noisy_stem shape (B, 4, 16, 54).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        base = cfg["base_channels"]                              # 64
        mults = cfg["channel_multipliers"]                      # [1,2,4,8]
        heads = cfg["attention_heads"]                           # 8
        t_dim = cfg["timestep_embed_dim"]                        # 256
        clap_dim = 512                                           # CLAP output dim
        cond_dim = t_dim + clap_dim                              # 768

        channels = [base * m for m in mults]                    # [64,128,256,512]

        # Input projection (noisy stem latent has 4 channels)
        self.stem_in_proj = nn.Conv2d(4, channels[0], 3, padding=1)
        self.mix_in_proj = nn.Conv2d(4, channels[0], 3, padding=1)

        # Timestep embedding
        self.t_emb = TimestepEmbedding(t_dim)

        # CLAP projection (512 -> t_dim to concat cleanly; kept at 512 width)
        # cond vector = cat(t_emb, clap_emb) so cond_dim = t_dim + 512
        # FiLM uses cond_dim directly.

        # Encoders
        self.stem_enc = nn.ModuleList()
        self.mix_enc = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.stem_enc.append(EncoderLevel(in_ch, out_ch, cond_dim, use_film=True))
            self.mix_enc.append(EncoderLevel(in_ch, out_ch, cond_dim, use_film=False))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = BottleneckAttention(channels[-1], heads)

        # Decoder
        # Skips from stem encoder levels [1..3] in reverse, plus the initial proj
        skip_channels = [channels[0]] + list(channels[1:])      # [64,128,256,512]
        dec_in_channels = list(reversed(channels))               # [512,256,128,64]
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            d_in = dec_in_channels[i]
            skip_ch = d_in                  # enc skip at same depth has same channel count
            d_out = dec_in_channels[i + 1]
            self.decoder.append(DecoderLevel(d_in, skip_ch, d_out, cond_dim))

        # Final conv: predict noise for all 4 latent channels
        self.final_conv = nn.Conv2d(channels[0] + 4, 4, 1)

    def forward(
        self,
        noisy_stem: torch.Tensor,
        mixture: torch.Tensor,
        clap_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_stem: (B, 4, F, T) noisy stem latent
            mixture:    (B, 4, F, T) clean mixture latent
            clap_emb:   (B, 512) CLAP conditioning
            t:          (B,) integer timesteps
        Returns:
            predicted noise (B, 1, F, T) — matches noisy_stem spatial dims
        """
        t_emb = self.t_emb(t)                          # (B, t_dim)
        cond = torch.cat([t_emb, clap_emb], dim=-1)   # (B, cond_dim)

        # Input projections
        s = self.stem_in_proj(noisy_stem)              # (B, C0, F, T)
        m = self.mix_in_proj(mixture)                  # (B, C0, F, T)

        # Encoder forward passes
        stem_skips = [s]    # skip at level 0 = input proj output
        for stem_level, mix_level in zip(self.stem_enc, self.mix_enc):
            s, s_skip = stem_level(s, cond)
            m, _ = mix_level(m, None)
            stem_skips.append(s_skip)
        # stem_skips: [C0_feat, C1_feat, C2_feat, C3_feat] (before pooling)

        # Bottleneck
        s = self.bottleneck(s, m)

        # Decoder
        for i, dec_level in enumerate(self.decoder):
            skip = stem_skips[-(i + 1)]
            s = dec_level(s, skip, cond)

        # Final skip with original noisy input
        s = torch.cat([s, noisy_stem], dim=1)
        return self.final_conv(s)                      # (B, 1, F, T)
