"""Dual-encoder UNet for query-conditioned stem separation.

Stable Audio Open VAE produces 1-D latents (B, 64, T_lat).  This UNet uses a
"1D-in-2D" adapter: the latent is unsqueezed to (B, 64, 1, T_lat) so the
existing Conv2d / GroupNorm blocks work unchanged.  Pooling and upsampling are
restricted to the time axis only (kernel (1,2), stride (1,2)) to avoid
collapsing the height dimension from 1 to 0.

TODO (long-term): convert all blocks to Conv1d for a cleaner architecture.
"""
import torch
import torch.nn as nn

from models.conv_blocks import ConvBlock, FiLM, TimestepEmbedding
from models.attention import BottleneckAttention

LATENT_CH = 64   # Stable Audio Open VAE latent channels


class EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, use_film: bool):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.film = FiLM(cond_dim, out_ch) if use_film else None
        # Time-only pooling: height stays 1, width (time) halves.
        self.pool = nn.MaxPool2d((1, 2))

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
        # Time-only upsampling to match the time-only pooling in the encoder.
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=(1, 2), stride=(1, 2))
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
        self.film = FiLM(cond_dim, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        # Align time dim if odd sizes arise from non-power-of-2 T_lat.
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

    Input latents are (B, 64, T_lat) from the Stable Audio Open VAE.
    They are unsqueezed to (B, 64, 1, T_lat) internally and squeezed back at output.
    Output: predicted noise (B, 64, T_lat) matching noisy_stem shape.
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

        # Input projections: each takes one 64-channel SAO latent.
        self.stem_in_proj = nn.Conv2d(LATENT_CH, channels[0], 3, padding=1)
        self.mix_in_proj  = nn.Conv2d(LATENT_CH, channels[0], 3, padding=1)

        # Timestep embedding
        self.t_emb = TimestepEmbedding(t_dim)

        # Encoders (3 levels: channels[1:] = [128, 256, 512])
        self.stem_enc = nn.ModuleList()
        self.mix_enc  = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.stem_enc.append(EncoderLevel(in_ch, out_ch, cond_dim, use_film=True))
            self.mix_enc.append(EncoderLevel(in_ch, out_ch, cond_dim, use_film=False))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = BottleneckAttention(channels[-1], heads)

        # Decoder (skips from stem encoder levels [1..3] in reverse, plus initial proj)
        dec_in_channels = list(reversed(channels))               # [512,256,128,64]
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            d_in   = dec_in_channels[i]
            d_out  = dec_in_channels[i + 1]
            self.decoder.append(DecoderLevel(d_in, d_in, d_out, cond_dim))

        # Final conv: predict noise at full 64-channel latent resolution.
        self.final_conv = nn.Conv2d(channels[0] + LATENT_CH, LATENT_CH, 1)

    def forward(
        self,
        noisy_stem: torch.Tensor,
        mixture: torch.Tensor,
        clap_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_stem: (B, 64, T_lat) noisy stem latent
            mixture:    (B, 64, T_lat) clean mixture latent
            clap_emb:   (B, 512) CLAP conditioning
            t:          (B,) integer timesteps
        Returns:
            predicted noise (B, 64, T_lat) — matches noisy_stem shape
        """
        # Adapter: treat 1-D latent as 2-D with a single "frequency" row.
        noisy_stem = noisy_stem.unsqueeze(2)   # (B, 64, 1, T_lat)
        mixture    = mixture.unsqueeze(2)      # (B, 64, 1, T_lat)

        t_emb = self.t_emb(t)                          # (B, t_dim)
        cond  = torch.cat([t_emb, clap_emb], dim=-1)   # (B, cond_dim)

        # Input projections
        s = self.stem_in_proj(noisy_stem)              # (B, C0, 1, T_lat)
        m = self.mix_in_proj(mixture)                  # (B, C0, 1, T_lat)

        # Encoder forward passes
        stem_skips = [s]
        for stem_level, mix_level in zip(self.stem_enc, self.mix_enc):
            s, s_skip = stem_level(s, cond)
            m, _      = mix_level(m, None)
            stem_skips.append(s_skip)

        # Bottleneck — operates on (B, C, 1, T) flattened to N=T tokens
        s = self.bottleneck(s, m)

        # Decoder
        for i, dec_level in enumerate(self.decoder):
            skip = stem_skips[-(i + 1)]
            s = dec_level(s, skip, cond)

        # Final skip connection with original noisy input
        s = torch.cat([s, noisy_stem], dim=1)          # (B, C0+64, 1, T_lat)
        out = self.final_conv(s)                        # (B, 64, 1, T_lat)

        return out.squeeze(2)                           # (B, 64, T_lat)
