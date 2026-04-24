"""Wiener filter, temporal smoothing, residual constraint, and OLA helpers."""
import torch
import torch.nn.functional as F
import numpy as np


def wiener_filter(
    predicted_stems: list[torch.Tensor],
    mixture_stft: torch.Tensor,
    eps: float = 1e-8,
) -> list[torch.Tensor]:
    """
    Soft Wiener mask over STFT magnitudes.

    Args:
        predicted_stems: list of N (B, 1, F, T) waveforms (time domain)
        mixture_stft:    (B, 1+F, T) complex STFT of the mixture
    Returns:
        list of N Wiener-filtered waveforms (B, 1, T)
    """
    if len(predicted_stems) < 2:
        return predicted_stems

    energies = [s.pow(2).mean(dim=1, keepdim=True) for s in predicted_stems]
    total_energy = sum(energies).clamp(min=eps)

    filtered = []
    for stem, energy in zip(predicted_stems, energies):
        mask = energy / total_energy   # (B, 1, 1, 1) broadcast
        filtered.append(stem * mask)
    return filtered


def temporal_smooth(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Gaussian smoothing along the last (time) dimension.

    Args:
        mask: (..., T)
    Returns:
        smoothed mask same shape
    """
    sigma = kernel_size / 3.0
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=mask.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    orig_shape = mask.shape
    flat = mask.reshape(-1, 1, mask.shape[-1])
    pad = (half, half)
    flat_pad = F.pad(flat, pad, mode="reflect")
    kernel = kernel.view(1, 1, -1)
    smoothed = F.conv1d(flat_pad, kernel)
    return smoothed.reshape(orig_shape)


def residual_constraint(
    predicted_stem: torch.Tensor,
    mixture: torch.Tensor,
    max_suppression_db: float = -20.0,
) -> torch.Tensor:
    """
    Clamp predicted stem energy relative to mixture to avoid hallucination.

    Args:
        predicted_stem: (B, 1, T)
        mixture:        (B, 1, T)
        max_suppression_db: floor at this many dB below mixture
    Returns:
        constrained stem (B, 1, T)
    """
    floor_factor = 10 ** (max_suppression_db / 20.0)
    mix_rms = mixture.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    stem_rms = predicted_stem.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    floor_rms = floor_factor * mix_rms
    ratio = (floor_rms / stem_rms).clamp(max=1.0)
    return predicted_stem * ratio
