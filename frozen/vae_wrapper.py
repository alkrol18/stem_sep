"""Stable Audio Open waveform-domain VAE encode/decode interface (frozen, 44.1 kHz).

Replaces the AudioLDM2 mel-spectrogram VAE.  The Stable Audio Open pretransform
(AutoencoderOobleck) operates on raw stereo waveforms and produces 64-channel
latents at ~21.5 Hz (2048x compression at 44.1 kHz).  No vocoder is needed.

HuggingFace access: stabilityai/stable-audio-open-1.0 is a gated model.
Before loading, set your token:
    export HF_TOKEN=hf_...
or log in with `huggingface-cli login`.
"""
import os
import torch
import torch.nn as nn


class VAEWrapper(nn.Module):
    def __init__(self, checkpoint: str = "stabilityai/stable-audio-open-1.0"):
        super().__init__()
        from stable_audio_tools import get_pretrained_model

        # Load the full pipeline to extract just the pretransform (VAE).
        # The rest of the diffusion model is deleted immediately to save VRAM.
        full_model, self._model_config = get_pretrained_model(checkpoint)
        self.vae = full_model.pretransform
        del full_model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.vae.requires_grad_(False)
        self.vae.eval()

    # ------------------------------------------------------------------
    # Device management: forward to the inner module since nn.Module.to()
    # won't recurse into attributes that aren't registered submodules.
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        self.vae = self.vae.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode mono waveform to latent.

        Args:
            waveform: (B, 1, T) mono float32 at 44.1 kHz
        Returns:
            latent: (B, 64, T_lat)  where T_lat = T / 2048
        """
        # Oobleck VAE expects stereo (2-channel) input.
        stereo = torch.cat([waveform, waveform], dim=1)   # (B, 2, T)

        result = self.vae.encode(stereo)

        # stable-audio-tools ≥1.0 returns a plain tensor; older builds return
        # (latent, info_dict).  Handle both gracefully.
        if isinstance(result, (tuple, list)):
            latent = result[0]
        elif hasattr(result, "z"):          # EncoderOutput dataclass
            latent = result.z
        else:
            latent = result

        return latent   # (B, 64, T_lat)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to mono waveform.

        Args:
            latent: (B, 64, T_lat)
        Returns:
            waveform: (B, 1, T) mono float32 at 44.1 kHz
        """
        stereo = self.vae.decode(latent)                  # (B, 2, T)
        return stereo.mean(dim=1, keepdim=True)           # (B, 1, T)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))
