"""AudioLDM2 VAE encode/decode interface (frozen)."""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    def __init__(self, checkpoint: str = "cvssp/audioldm2"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        # Scaling factor from the VAE config (matches AudioLDM2 training)
        self.scale = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, n_mels, T) mel spectrogram
        Returns:
            latent: (B, 4, latent_freq, latent_time)
        """
        dist = self.vae.encode(mel).latent_dist
        return dist.sample() * self.scale

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 4, latent_freq, latent_time)
        Returns:
            mel: (B, 1, n_mels, T)
        """
        return self.vae.decode(latent / self.scale).sample

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(mel))
