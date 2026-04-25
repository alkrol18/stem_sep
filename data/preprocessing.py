"""Mel spectrogram helpers, chunking, and overlap-add reconstruction."""
import math
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def to_mel(waveform: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Args:
        waveform: (1, T) mono float32
        config:   audio sub-dict from default.yaml
    Returns:
        mel: (1, n_mels, T_frames) log1p-scaled
    """
    transform = T.MelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        power=2.0,
    ).to(waveform.device)
    mel = transform(waveform)               # (1, n_mels, T_frames)
    return torch.log1p(mel)


def from_mel(mel: torch.Tensor, config: dict) -> torch.Tensor:
    """Approximate inversion for verification only (Griffin-Lim via torchaudio)."""
    mel_lin = torch.expm1(mel.clamp(min=0))
    inv = T.InverseMelScale(
        n_stft=config["n_fft"] // 2 + 1,
        n_mels=config["n_mels"],
        sample_rate=config["sample_rate"],
    ).to(mel.device)
    griffin = T.GriffinLim(
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    ).to(mel.device)
    spec = inv(mel_lin.squeeze(0))          # (n_stft, T_frames)
    wav = griffin(spec)                     # (T,)
    return wav.unsqueeze(0)                 # (1, T)


def _hann_window(length: int, device: torch.device) -> torch.Tensor:
    return torch.hann_window(length, device=device)


def chunk_with_hann(
    waveform: torch.Tensor,
    chunk_samples: int,
    hop_samples: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Splits waveform into Hann-windowed chunks.

    Args:
        waveform:      (1, T)
        chunk_samples: window length
        hop_samples:   hop between windows
    Returns:
        chunks:    list of (1, chunk_samples) tensors (zero-padded at boundaries)
        positions: list of start sample indices
    """
    T = waveform.shape[-1]
    win = _hann_window(chunk_samples, waveform.device)  # (chunk_samples,)
    chunks, positions = [], []

    start = 0
    while start < T:
        end = start + chunk_samples
        if end <= T:
            chunk = waveform[..., start:end]
        else:
            pad = end - T
            chunk = torch.nn.functional.pad(waveform[..., start:T], (0, pad))
        chunk = chunk * win.unsqueeze(0)
        chunks.append(chunk)
        positions.append(start)
        start += hop_samples

    return chunks, positions


def ola_reconstruct(
    chunks: list[torch.Tensor],
    positions: list[int],
    window: torch.Tensor,
    total_length: int,
) -> torch.Tensor:
    """
    Overlap-add reconstruction with normalisation by summed window energy.

    Args:
        chunks:       list of (1, chunk_samples) tensors
        positions:    list of start sample indices
        window:       (chunk_samples,) Hann window
        total_length: target output length
    Returns:
        (1, total_length) reconstructed waveform
    """
    device = chunks[0].device
    out = torch.zeros(1, total_length, device=device)
    win_sum = torch.zeros(total_length, device=device)
    win2 = window ** 2

    for chunk, pos in zip(chunks, positions):
        length = min(chunk.shape[-1], total_length - pos)
        out[..., pos:pos + length] += (chunk * window.unsqueeze(0))[..., :length]
        win_sum[pos:pos + length] += win2[:length]

    win_sum = win_sum.clamp(min=1e-8)
    return out / win_sum.unsqueeze(0)
