"""HiFi-GAN vocoder interface, matched to AudioLDM2 mel config (frozen)."""
import torch
import torch.nn as nn
from diffusers import AudioLDM2Pipeline


class VocoderWrapper(nn.Module):
    """
    Loads the HiFi-GAN vocoder from AudioLDM2's diffusers pipeline so the mel
    configuration (n_mels=128, sr=16 kHz internal, then resampled) matches exactly.
    """

    def __init__(self, checkpoint: str = "cvssp/audioldm2"):
        super().__init__()
        pipe = AudioLDM2Pipeline.from_pretrained(checkpoint, torch_dtype=torch.float32)
        self.vocoder = pipe.vocoder
        self.vocoder.requires_grad_(False)
        self.vocoder.eval()
        # AudioLDM2 vocoder works at 16 kHz internally; we resample to 44.1 kHz
        self.internal_sr = 16000
        self.target_sr = 44100

    def to(self, device):
        self.vocoder = self.vocoder.to(device)
        return super().to(device)

    @torch.no_grad()
    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, n_mels, T) or (B, n_mels, T) log-mel spectrogram
        Returns:
            waveform: (B, 1, T_audio) at 44100 Hz
        """
        import torchaudio

        if mel.dim() == 4:
            mel = mel.squeeze(1)  # (B, n_mels, T)

        wav = self.vocoder(mel)  # (B, T_16k)

        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # (B, 1, T_16k)

        resamp = torchaudio.transforms.Resample(
            self.internal_sr, self.target_sr
        ).to(wav.device)
        return resamp(wav)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.mel_to_wav(mel)
