"""Evaluate SDR/SIR/SAR on MUSDB18-HQ test split via mir_eval."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(checkpoint: str, musdb_root: str, config_path: str = "configs/default.yaml"):
    import musdb
    import mir_eval
    import torchaudio

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from frozen.vae_wrapper import VAEWrapper
    from frozen.clap_wrapper import CLAPWrapper
    from frozen.vocoder import VocoderWrapper
    from models.unet import StemSeparationUNet
    from models.diffusion import NoiseSchedule, DDIMSampler
    from training.checkpoint import load_checkpoint
    from data.preprocessing import to_mel, chunk_with_hann, ola_reconstruct
    from inference.postprocess import residual_constraint

    model = StemSeparationUNet(cfg["unet"]).to(device)
    noise_schedule = NoiseSchedule(cfg["diffusion"]).to(device)
    sampler = DDIMSampler(noise_schedule, cfg["diffusion"]["inference_steps"])
    load_checkpoint(checkpoint, model, device=str(device))
    model.eval()

    vae = VAEWrapper(cfg["vae"]["checkpoint"]).to(device)
    clap = CLAPWrapper(cfg["clap"]["checkpoint"], device=str(device)).to(device)
    vocoder = VocoderWrapper(cfg["vae"]["checkpoint"]).to(device)

    db = musdb.DB(root=musdb_root, is_wav=True, subsets=["test"])
    stem_names = ["vocals", "drums", "bass", "other"]

    results = {s: {"sdr": [], "sir": [], "sar": []} for s in stem_names}
    sr_target = cfg["audio"]["sample_rate"]
    chunk_samples = cfg["audio"]["chunk_samples"]
    hop_samples = cfg["audio"]["hop_samples"]
    window = torch.hann_window(chunk_samples)

    for track in db:
        print(f"  evaluating: {track.name}")
        mix_stereo = track.audio.T          # (2, T) numpy float64
        mix_mono = mix_stereo.mean(0).astype(np.float32)
        mix_wav = torch.from_numpy(mix_mono).unsqueeze(0)

        if track.rate != sr_target:
            mix_wav = torchaudio.functional.resample(mix_wav, track.rate, sr_target)

        for stem_name in stem_names:
            ref_stereo = track.targets[stem_name].audio.T
            ref_mono = ref_stereo.mean(0).astype(np.float32)

            query = f"isolate the {stem_name}"
            with torch.no_grad():
                clap_emb = clap.get_text_embedding([query])

            chunks, positions = chunk_with_hann(mix_wav, chunk_samples, hop_samples)
            stem_chunks = []
            for chunk in chunks:
                chunk = chunk.to(device)
                with torch.no_grad():
                    mel = to_mel(chunk.unsqueeze(0), cfg["audio"])
                    mix_latent = vae.encode(mel)
                    stem_latent = sampler.sample(
                        model, mix_latent, clap_emb.to(device), mix_latent.shape, device
                    )
                    stem_mel = vae.decode(stem_latent)
                    stem_wav_chunk = vocoder.mel_to_wav(stem_mel).squeeze(0)
                if stem_wav_chunk.shape[-1] >= chunk_samples:
                    stem_wav_chunk = stem_wav_chunk[..., :chunk_samples]
                else:
                    stem_wav_chunk = torch.nn.functional.pad(
                        stem_wav_chunk, (0, chunk_samples - stem_wav_chunk.shape[-1])
                    )
                stem_chunks.append(stem_wav_chunk.cpu())

            pred = ola_reconstruct(stem_chunks, positions, window, mix_wav.shape[-1])
            pred = residual_constraint(pred.unsqueeze(0), mix_wav.unsqueeze(0)).squeeze(0)
            pred_np = pred.squeeze(0).numpy()

            min_len = min(len(ref_mono), len(pred_np))
            try:
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                    ref_mono[:min_len][None], pred_np[:min_len][None]
                )
                results[stem_name]["sdr"].append(float(sdr[0]))
                results[stem_name]["sir"].append(float(sir[0]))
                results[stem_name]["sar"].append(float(sar[0]))
            except Exception as e:
                print(f"    [warn] {stem_name}: {e}")

    print("\n=== MUSDB18-HQ Test Results (median) ===")
    for stem_name in stem_names:
        sdr_vals = results[stem_name]["sdr"]
        sir_vals = results[stem_name]["sir"]
        sar_vals = results[stem_name]["sar"]
        if sdr_vals:
            print(
                f"  {stem_name:8s}  SDR={np.median(sdr_vals):.2f}  "
                f"SIR={np.median(sir_vals):.2f}  SAR={np.median(sar_vals):.2f}"
            )
        else:
            print(f"  {stem_name:8s}  no data")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--musdb_root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.musdb_root, args.config)
