"""BottleneckAttention: self-attention on stem tokens + cross-attention to mixture."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAttention(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        assert channels % heads == 0, "channels must be divisible by heads"

        # Self-attention on stem
        self.sa_norm = nn.LayerNorm(channels)
        self.sa_q = nn.Linear(channels, channels)
        self.sa_k = nn.Linear(channels, channels)
        self.sa_v = nn.Linear(channels, channels)
        self.sa_out = nn.Linear(channels, channels)

        # Cross-attention: stem queries mixture
        self.ca_norm_stem = nn.LayerNorm(channels)
        self.ca_norm_mix = nn.LayerNorm(channels)
        self.ca_q = nn.Linear(channels, channels)
        self.ca_k = nn.Linear(channels, channels)
        self.ca_v = nn.Linear(channels, channels)
        self.ca_out = nn.Linear(channels, channels)

        # Feed-forward
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def _mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        N_q: int,
        N_k: int,
        C: int,
    ) -> torch.Tensor:
        H, D = self.heads, self.head_dim
        q = q.view(B, N_q, H, D).transpose(1, 2)   # (B, H, N_q, D)
        k = k.view(B, N_k, H, D).transpose(1, 2)
        v = v.view(B, N_k, H, D).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, H, N_q, D)
        return attn.transpose(1, 2).contiguous().view(B, N_q, C)

    def forward(
        self,
        stem_feat: torch.Tensor,
        mix_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stem_feat: (B, C, F, T)
            mix_feat:  (B, C, F, T)
        Returns:
            (B, C, F, T) updated stem features
        """
        B, C, F, T = stem_feat.shape
        N = F * T

        # Flatten to sequence
        s = stem_feat.permute(0, 2, 3, 1).reshape(B, N, C)   # (B, N, C)
        m = mix_feat.permute(0, 2, 3, 1).reshape(B, N, C)

        # Self-attention
        s_norm = self.sa_norm(s)
        q = self.sa_q(s_norm)
        k = self.sa_k(s_norm)
        v = self.sa_v(s_norm)
        s = s + self.sa_out(self._mha(q, k, v, B, N, N, C))

        # Cross-attention
        s_norm = self.ca_norm_stem(s)
        m_norm = self.ca_norm_mix(m)
        q = self.ca_q(s_norm)
        k = self.ca_k(m_norm)
        v = self.ca_v(m_norm)
        s = s + self.ca_out(self._mha(q, k, v, B, N, N, C))

        # FFN
        s = s + self.ffn(self.ffn_norm(s))

        return s.reshape(B, F, T, C).permute(0, 3, 1, 2).contiguous()
