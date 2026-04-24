"""End-to-end inference pipeline: MP3/WAV in, separated stem WAV out."""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import to_mel, chunk_with_hann, ola_reconstruct
from frozen.vae_wrapper import VAEWrapper
from frozen.clap_wrapper import CLAPWrapper
from frozen.vocoder import VocoderWrapper
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
    """
    Separate a single (1, chunk_samples) chunk.
    Returns (1, chunk_samples) predicted stem waveform.
    """
    from frozen.vocoder import VocoderWrapper

    chunk_wav = chunk_wav.unsqueeze(0).to(device)          # (1, 1, T)
    with torch.no_grad():
        mel = to_mel(chunk_wav, cfg["audio"])               # (1, 1, n_mels, T_mel)
        mix_latent = vae.encode(mel)                        # (1, 4, F, T)
        clap_emb = clap.get_text_embedding([query])         # (1, 512)

        shape = mix_latent.shape
        stem_latent = sampler.sample(
            model, mix_latent, clap_emb, shape, device
        )
        stem_mel = vae.decode(stem_latent)                  # (1, 1, n_mels, T_mel)

    return stem_mel.squeeze(0)  # (1, n_mels, T_mel) — vocoder handles this


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
    model = StemSeparationUNet(cfg["unet"]).to(device)
    noise_schedule = NoiseSchedule(cfg["diffusion"]).to(device)
    sampler = DDIMSampler(noise_schedule, cfg["diffusion"]["inference_steps"])
    load_checkpoint(checkpoint, model, device=str(device))
    model.eval()

    # Frozen encoders
    vae = VAEWrapper(cfg["vae"]["checkpoint"]).to(device)
    clap = CLAPWrapper(cfg["clap"]["checkpoint"], device=str(device)).to(device)
    vocoder = VocoderWrapper(cfg["vae"]["checkpoint"]).to(device)

    # Load and decode input audio to mono 44.1 kHz
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
    hop_samples = cfg["audio"]["hop_samples"]

    chunks, positions = chunk_with_hann(wav, chunk_samples, hop_samples)
    window = _hann_window(chunk_samples, device=device)

    stem_chunks = []
    for chunk in chunks:
        chunk = chunk.to(device)
        with torch.no_grad():
            mel = to_mel(chunk.unsqueeze(0), cfg["audio"])      # (1, 1, n_mels, T)
            mix_latent = vae.encode(mel)                         # (1, 4, F, T)
            shape = mix_latent.shape
            stem_latent = sampler.sample(
                model, mix_latent, query_emb.expand(shape[0], -1), shape, device
            )
            stem_mel = vae.decode(stem_latent)                   # (1, 1, n_mels, T_mel)
            stem_wav = vocoder.mel_to_wav(stem_mel)              # (1, 1, T_wav)
            stem_wav = stem_wav.squeeze(0)                       # (1, T_wav)

        # Trim/pad back to chunk_samples at 44.1 kHz
        if stem_wav.shape[-1] >= chunk_samples:
            stem_wav = stem_wav[..., :chunk_samples]
        else:
            stem_wav = torch.nn.functional.pad(stem_wav, (0, chunk_samples - stem_wav.shape[-1]))

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
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--query", default="isolate the vocals")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ref_audio", default=None)
    args = parser.parse_args()
    separate(args.mixture, args.query, args.checkpoint, args.output, args.config, args.ref_audio)
