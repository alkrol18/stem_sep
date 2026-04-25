"""Sanity check for the Stable Audio Open VAE migration.

Run this interactively on the DCC before submitting a SLURM training job:

    module load cuda/12.1
    source /path/to/venv/bin/activate
    cd /home/users/ak724/stem_sep
    python scripts/sanity_check.py --config configs/dcc.yaml

Prerequisites:
    pip install stable-audio-tools einops
    export HF_TOKEN=hf_<your_token>   # model is gated on HuggingFace
    # Accept the licence at https://huggingface.co/stabilityai/stable-audio-open-1.0
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(cond: bool, msg: str):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {msg}")
    return cond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dcc.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"\n{INFO}  device = {device}")
    print(f"{INFO}  config = {args.config}")
    print()

    # ------------------------------------------------------------------ #
    # 1. VAE load and encode/decode shapes                                #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("1. VAE — load and shape verification")
    print("=" * 60)
    from frozen.vae_wrapper import VAEWrapper

    checkpoint = cfg["vae"]["checkpoint"]
    print(f"  Loading VAE from {checkpoint!r} ...")
    try:
        vae = VAEWrapper(checkpoint).to(device)
        print(f"  {PASS}  VAE loaded")
    except Exception as e:
        print(f"  {FAIL}  VAE load failed: {e}")
        print("\nTip: make sure HF_TOKEN is set and you have accepted the model licence.")
        sys.exit(1)

    sr      = cfg["audio"]["sample_rate"]            # 44100
    chunk   = cfg["audio"]["chunk_samples"]          # 221184
    T_lat_cfg = cfg["vae"]["latent_time"]            # expected 108
    lat_ch_cfg = cfg["vae"]["latent_channels"]       # expected 64

    # Encode a silent chunk
    dummy_wav = torch.zeros(1, 1, chunk, device=device)
    latent = vae.encode(dummy_wav)
    print()
    print(f"  Input waveform shape : {tuple(dummy_wav.shape)}  (B=1, C=1, T={chunk})")
    print(f"  Latent shape         : {tuple(latent.shape)}")
    lat_ch   = latent.shape[1]
    lat_time = latent.shape[2]
    compression = chunk / lat_time
    print(f"  Compression ratio    : {chunk} / {lat_time} = {compression:.1f}×")
    print(f"  Latent framerate     : {sr / compression:.2f} Hz")

    print()
    ok = check(latent.ndim == 3,       f"latent is 3-D (B, C, T)  — got {latent.ndim}-D")
    ok = check(lat_ch == lat_ch_cfg,   f"latent channels = {lat_ch}  (config says {lat_ch_cfg})")
    ok = check(lat_time == T_lat_cfg,  f"latent time     = {lat_time}  (config says {T_lat_cfg})")

    if lat_time != T_lat_cfg:
        print(f"\n  *** Update configs/dcc.yaml: vae.latent_time: {lat_time}")
        print(f"  *** Also update chunk_samples to {lat_time * int(compression)} "
              f"if you want a clean integer.\n")

    print(f"\n  Latent stats  mean={latent.mean().item():.4f}  "
          f"std={latent.std().item():.4f}  "
          f"min={latent.min().item():.4f}  "
          f"max={latent.max().item():.4f}")
    print("  (std should be near 1 for stable diffusion training; "
          "if far from 1, consider adding a scaling factor in vae_wrapper.py)")

    # Decode back
    recon = vae.decode(latent)
    print(f"\n  Decoded waveform shape: {tuple(recon.shape)}")
    ok = check(recon.shape == dummy_wav.shape,
               f"encode→decode round-trip preserves shape {tuple(dummy_wav.shape)}")

    # ------------------------------------------------------------------ #
    # 2. UNet single forward pass                                         #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("2. UNet — forward pass with actual latent shape")
    print("=" * 60)
    from models.unet import StemSeparationUNet
    from models.diffusion import NoiseSchedule

    model = StemSeparationUNet(cfg["unet"]).to(device)
    noise_schedule = NoiseSchedule(cfg["diffusion"]).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet parameters : {n_params:,}  ({n_params/1e6:.1f} M)")

    B        = 2
    clap_dim = 512
    noisy    = torch.randn(B, lat_ch, lat_time, device=device)
    mixture  = torch.randn(B, lat_ch, lat_time, device=device)
    clap_emb = torch.randn(B, clap_dim, device=device)
    t        = torch.randint(0, cfg["diffusion"]["num_timesteps"], (B,), device=device)

    try:
        with torch.no_grad():
            pred = model(noisy, mixture, clap_emb, t)
        print(f"  UNet input  shape : {tuple(noisy.shape)}")
        print(f"  UNet output shape : {tuple(pred.shape)}")
        ok = check(pred.shape == noisy.shape,
                   f"UNet output matches noisy_stem shape {tuple(noisy.shape)}")
    except Exception as e:
        print(f"  {FAIL}  UNet forward failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # 3. Noise schedule q_sample                                          #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("3. Noise schedule q_sample")
    print("=" * 60)
    x0 = torch.randn(B, lat_ch, lat_time, device=device)
    x_t, noise = noise_schedule.q_sample(x0, t)
    ok = check(x_t.shape == x0.shape, f"q_sample output shape {tuple(x_t.shape)}")

    # ------------------------------------------------------------------ #
    # 4. VRAM estimate                                                    #
    # ------------------------------------------------------------------ #
    if device.type == "cuda":
        print()
        print("=" * 60)
        print("4. VRAM usage")
        print("=" * 60)
        torch.cuda.synchronize()
        alloc_mb = torch.cuda.memory_allocated(device) / 1024**2
        reserv_mb = torch.cuda.memory_reserved(device) / 1024**2
        total_mb  = torch.cuda.get_device_properties(device).total_memory / 1024**2
        print(f"  Allocated : {alloc_mb:,.0f} MB")
        print(f"  Reserved  : {reserv_mb:,.0f} MB")
        print(f"  GPU total : {total_mb:,.0f} MB")
        headroom = total_mb - reserv_mb
        print(f"  Headroom  : {headroom:,.0f} MB  "
              f"({'OK for batch 16' if headroom > 8000 else 'tight — use batch 8'})")

    # ------------------------------------------------------------------ #
    # 5. Summary                                                          #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Summary of key values (update dcc.yaml if different)")
    print("=" * 60)
    print(f"  sample_rate        : {sr}")
    print(f"  chunk_samples      : {chunk}  ({chunk/sr:.3f} s)")
    print(f"  hop_samples        : {cfg['audio']['hop_samples']}")
    print(f"  vae.latent_channels: {lat_ch}")
    print(f"  vae.latent_time    : {lat_time}  "
          f"{'(matches config)' if lat_time == T_lat_cfg else f'(CONFIG SAYS {T_lat_cfg} — FIX NEEDED)'}")
    print(f"  compression_ratio  : {compression:.0f}x")
    print(f"  unet n_params      : {n_params/1e6:.1f} M")
    print()


if __name__ == "__main__":
    main()
