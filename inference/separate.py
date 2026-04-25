"""End-to-end inference pipeline: MP3/WAV in, separated stem WAV out.

With the Stable Audio Open VAE, the pipeline is:
    waveform → vae.encode → latent → diffusion sampler → vae.decode → waveform
No vocoder or mel spectrogram is needed.  Output is native 44.1 kHz stereo
averaged to mono.
"""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import chunk_with_hann, ola_reconstruct
from frozen.vae_wrapper import VAEWrapper
from frozen.clap_wrapper import CLAPWrapper
from models.unet import StemSeparationUNet
from models.diffusion import NoiseSchedule, DDIMSampler
from inference.postprocess import residual_constraint
from training.checkpoint import load_checkpoint


def _hann_window(length: int, device) -> torch.Tensor:
    return torch.hann_window(length, device=device)


def _separate_chunk(
    chunk_wav: torch.Tensor,
    query: str,
    model,
    vae: VAEWrapper,
    clap: CLAPWrapper,
    noise_schedule: NoiseSchedule,
    sampler: DDIMSampler,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    """Separate a single (1, T) chunk waveform.

    Returns (1, T) predicted stem waveform at 44.1 kHz.
    """
    chunk_wav = chunk_wav.unsqueeze(0).to(device)          # (1, 1, T)
    with torch.no_grad():
        mix_latent = vae.encode(chunk_wav)                 # (1, 64, T_lat)
        clap_emb   = clap.get_text_embedding([query])      # (1, 512)
        shape      = mix_latent.shape
        stem_latent = sampler.sample(
            model, mix_latent, clap_emb, shape, device
        )                                                  # (1, 64, T_lat)
        stem_wav = vae.decode(stem_latent)                 # (1, 1, T)

    return stem_wav.squeeze(0)                             # (1, T)


def separate(
    mixture_path: str,
    query: str,
    checkpoint: str,
    output_path: str,
    config_path: str = "configs/default.yaml",
    ref_audio: str | None = None,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model          = StemSeparationUNet(cfg["unet"]).to(device)
    noise_schedule = NoiseSchedule(cfg["diffusion"]).to(device)
    sampler        = DDIMSampler(noise_schedule, cfg["diffusion"]["inference_steps"])
    load_checkpoint(checkpoint, model, device=str(device))
    model.eval()

    # Frozen encoders (no vocoder needed)
    vae  = VAEWrapper(cfg["vae"]["checkpoint"]).to(device)
    clap = CLAPWrapper(cfg["clap"]["checkpoint"], device=str(device)).to(device)

    # Load input audio — target 44.1 kHz mono
    sr_target = cfg["audio"]["sample_rate"]
    wav, sr = torchaudio.load(mixture_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)

    # Build query embedding (reference audio overrides text query)
    if ref_audio:
        ref_wav, ref_sr = torchaudio.load(ref_audio)
        if ref_wav.shape[0] > 1:
            ref_wav = ref_wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            query_emb = clap.get_audio_embedding(ref_wav.to(device), sr=ref_sr)
    else:
        with torch.no_grad():
            query_emb = clap.get_text_embedding([query])   # (1, 512)

    chunk_samples = cfg["audio"]["chunk_samples"]
    hop_samples   = cfg["audio"]["hop_samples"]

    chunks, positions = chunk_with_hann(wav, chunk_samples, hop_samples)
    window = _hann_window(chunk_samples, device=device)

    stem_chunks = []
    for chunk in chunks:
        chunk = chunk.to(device)                            # (1, T)
        with torch.no_grad():
            mix_latent  = vae.encode(chunk.unsqueeze(0))   # (1, 64, T_lat)
            shape       = mix_latent.shape
            stem_latent = sampler.sample(
                model, mix_latent, query_emb.expand(shape[0], -1), shape, device
            )
            stem_wav = vae.decode(stem_latent)             # (1, 1, T)
            stem_wav = stem_wav.squeeze(0)                 # (1, T)

        # Trim/pad back to chunk_samples
        if stem_wav.shape[-1] >= chunk_samples:
            stem_wav = stem_wav[..., :chunk_samples]
        else:
            stem_wav = torch.nn.functional.pad(
                stem_wav, (0, chunk_samples - stem_wav.shape[-1])
            )
        stem_chunks.append(stem_wav.cpu())

    total_length = wav.shape[-1]
    stem_out = ola_reconstruct(stem_chunks, positions, window.cpu(), total_length)

    # Residual constraint
    stem_out = residual_constraint(stem_out.unsqueeze(0), wav.unsqueeze(0)).squeeze(0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, stem_out.squeeze(0).numpy(), sr_target)
    print(f"[separate] saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture",    required=True)
    parser.add_argument("--query",      default="isolate the vocals")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--ref_audio",  default=None)
    args = parser.parse_args()
    separate(args.mixture, args.query, args.checkpoint, args.output, args.config, args.ref_audio)
