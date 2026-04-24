"""LAION-CLAP text/audio embedding interface (frozen)."""
import torch
import torch.nn as nn
import numpy as np


class CLAPWrapper(nn.Module):
    def __init__(self, checkpoint: str, device: str = "cpu"):
        super().__init__()
        import laion_clap
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt(checkpoint)
        self.model.requires_grad_(False)
        self.model.eval()
        self._device = device

    def to(self, device):
        self._device = str(device)
        self.model = self.model.to(device)
        return super().to(device)

    @torch.no_grad()
    def get_text_embedding(self, text_list: list[str]) -> torch.Tensor:
        """
        Args:
            text_list: list of B query strings
        Returns:
            (B, 512) float32 tensor on self._device
        """
        emb = self.model.get_text_embedding(text_list, use_tensor=True)
        return emb.to(self._device)

    @torch.no_grad()
    def get_audio_embedding(self, waveform: torch.Tensor, sr: int = 44100) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) mono float32 tensor at sr
        Returns:
            (B, 512) float32 tensor on self._device
        """
        # CLAP expects 48 kHz; resample if needed
        if sr != 48000:
            import torchaudio
            resamp = torchaudio.transforms.Resample(sr, 48000).to(waveform.device)
            waveform = resamp(waveform)
        emb = self.model.get_audio_embedding_from_data(waveform, use_tensor=True)
        return emb.to(self._device)

    def forward(self, text_list: list[str]) -> torch.Tensor:
        return self.get_text_embedding(text_list)
