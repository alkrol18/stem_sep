"""Main training loop for query-conditioned stem separation."""
import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MoisesDataset
from frozen.vae_wrapper import VAEWrapper
from frozen.clap_wrapper import CLAPWrapper
from models.unet import StemSeparationUNet
from models.diffusion import NoiseSchedule, DDIMSampler
from training.losses import diffusion_mse_loss, smoothness_loss, consistency_loss
from training.checkpoint import save_checkpoint, load_checkpoint


def build_lr_scheduler(optimizer, cfg, total_steps: int):
    warmup_steps = cfg["warmup_steps"]
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def eval_musdb(model, vae, clap, noise_schedule, sampler, cfg, device):
    """Evaluate on MUSDB18-HQ validation subset (SDR). Returns dict of metrics."""
    musdb_root = cfg["paths"]["musdb_root"]
    if not Path(musdb_root).exists():
        return {}
    try:
        import musdb
        import mir_eval
        import numpy as np
        from inference.separate import _separate_chunk

        db = musdb.DB(root=musdb_root, is_wav=True, subsets=["test"])
        stem_names = ["vocals", "drums", "bass", "other"]
        all_sdr = {s: [] for s in stem_names}

        model.eval()
        for track in db[:5]:   # quick eval on first 5 tracks
            mix_audio = track.audio.T.mean(axis=0)  # mono (T,)
            for stem_name in stem_names:
                ref = track.targets[stem_name].audio.T.mean(axis=0)
                query = f"isolate the {stem_name}"
                with torch.no_grad():
                    pred = _separate_chunk(
                        torch.from_numpy(mix_audio).float().unsqueeze(0).to(device),
                        query, model, vae, clap, noise_schedule, sampler, cfg, device,
                    )
                # pred: (1, T_wav) waveform; ref: (T_wav,) numpy
                min_len = min(len(ref), pred.shape[-1])
                sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
                    ref[:min_len][None], pred[0, :min_len].cpu().numpy()[None]
                )
                all_sdr[stem_name].append(float(sdr[0]))

        model.train()
        return {k: float(sum(v) / len(v)) if v else 0.0 for k, v in all_sdr.items()}
    except Exception as e:
        print(f"[eval] skipped: {e}")
        return {}


def run_validation(model, val_iter, val_loader, vae, clap, noise_schedule, cfg, device, num_batches):
    """Quick val loss over `num_batches` batches. Returns (mean_loss, val_iter)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            mix_wav = batch["mixture_wav"].to(device)      # (B, 1, T)
            tgt_wav = batch["target_stem_wav"].to(device)  # (B, 1, T)
            queries = batch["query_text"]
            mix_latent = vae.encode(mix_wav)               # (B, 64, T_lat)
            tgt_latent = vae.encode(tgt_wav)               # (B, 64, T_lat)
            clap_emb = clap.get_text_embedding(list(queries))
            B = tgt_latent.shape[0]
            t = torch.randint(0, cfg["diffusion"]["num_timesteps"], (B,), device=device)
            x_t, noise = noise_schedule.q_sample(tgt_latent, t)
            pred_noise = model(x_t, mix_latent, clap_emb, t)
            loss = diffusion_mse_loss(pred_noise, noise)
            losses.append(loss.item())
    model.train()
    return (sum(losses) / max(1, len(losses))), val_iter


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # Determine LR for mode C
    lr = cfg["training"]["lr"]
    if args.mode == "c":
        lr = 2e-5

    # Data — train/val split
    index_path = str(Path(cfg["paths"]["moisesdb_root"]) / "index.json")
    train_dataset = MoisesDataset(index_path, cfg, augment=True)
    val_dataset   = MoisesDataset(index_path, cfg, augment=False)
    n       = len(train_dataset)
    val_frac = cfg["training"].get("val_frac", 0.1)
    n_val   = max(1, int(n * val_frac))
    g       = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=g).tolist()
    dataset    = Subset(train_dataset, indices[:n - n_val])
    val_subset = Subset(val_dataset,   indices[n - n_val:])
    loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=12, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_iter = iter(val_loader)

    # Frozen encoders
    vae  = VAEWrapper(cfg["vae"]["checkpoint"]).to(device)
    clap = CLAPWrapper(cfg["clap"]["checkpoint"], device=str(device))
    clap = clap.to(device)

    # Model
    model          = StemSeparationUNet(cfg["unet"]).to(device)
    noise_schedule = NoiseSchedule(cfg["diffusion"]).to(device)
    sampler        = DDIMSampler(noise_schedule, cfg["diffusion"]["inference_steps"])

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"]
    )
    total_steps = len(loader) * cfg["training"]["epochs"]
    scheduler   = build_lr_scheduler(optimizer, cfg["training"], total_steps)

    start_step, start_epoch = 0, 0
    if args.mode == "c" and args.init_checkpoint:
        start_step, start_epoch, _ = load_checkpoint(
            args.init_checkpoint, model, optimizer, scheduler, device=str(device)
        )
        print(f"[train] resumed from {args.init_checkpoint} (step={start_step})")

    writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])
    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    step = start_step
    model.train()

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        loop = tqdm(loader, desc=f"epoch {epoch}")
        for batch in loop:
            mix_wav = batch["mixture_wav"].to(device)       # (B, 1, T)
            tgt_wav = batch["target_stem_wav"].to(device)   # (B, 1, T)
            queries = batch["query_text"]                   # list of B strings

            # Waveform → VAE latent (frozen, augmentation already in dataset)
            with torch.no_grad():
                mix_latent = vae.encode(mix_wav)            # (B, 64, T_lat)
                tgt_latent = vae.encode(tgt_wav)            # (B, 64, T_lat)
                clap_emb   = clap.get_text_embedding(list(queries))

            # Forward diffusion
            B = tgt_latent.shape[0]
            t = torch.randint(0, cfg["diffusion"]["num_timesteps"], (B,), device=device)
            x_t, noise = noise_schedule.q_sample(tgt_latent, t)

            # UNet forward
            pred_noise = model(x_t, mix_latent, clap_emb, t)

            # Losses
            loss = diffusion_mse_loss(pred_noise, noise)

            if args.mode == "c":
                cfg_l = cfg["losses"]
                # Smoothness
                x0_pred = noise_schedule.predict_x0_from_noise(x_t, t, pred_noise)
                loss = loss + cfg_l["smoothness_weight"] * smoothness_loss(x0_pred)

                # Consistency (stochastic)
                if (
                    random.random() < cfg_l["consistency_batch_prob"]
                    and t.float().mean().item() < cfg_l["consistency_max_t"]
                ):
                    vocab = train_dataset.stem_vocab
                    k     = min(cfg_l["consistency_stem_cap"], len(vocab))
                    sampled_names = random.sample(vocab, k)
                    stem_queries = []
                    with torch.no_grad():
                        for sn in sampled_names:
                            text = [f"isolate the {sn}"] * B
                            emb  = clap.get_text_embedding(text)
                            stem_queries.append((emb, sn))
                    c_loss = consistency_loss(
                        model, mix_latent, stem_queries, noise_schedule, t, cfg_l
                    )
                    loss = loss + cfg_l["consistency_weight"] * c_loss
                    writer.add_scalar("loss/consistency", c_loss.item(), step)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
            loop.set_postfix(loss=f"{loss.item():.4f}")
            if step % cfg["training"].get("print_every_steps", 50) == 0:
                print(f"[train] step={step} epoch={epoch} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            if step > 0 and step % cfg["training"].get("val_every_steps", 200) == 0:
                val_loss, val_iter = run_validation(
                    model, val_iter, val_loader, vae, clap, noise_schedule, cfg, device,
                    cfg["training"].get("val_batches", 4),
                )
                writer.add_scalar("loss/val", val_loss, step)
                print(f"[val] step={step} val_loss={val_loss:.4f}", flush=True)
            step += 1

        # Checkpoint
        if (epoch + 1) % cfg["training"]["checkpoint_every"] == 0:
            ckpt_path = str(Path(cfg["paths"]["checkpoint_dir"]) / f"ckpt_epoch{epoch+1}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step, epoch + 1, cfg)
            print(f"[train] saved {ckpt_path}")

        # Eval
        if (epoch + 1) % cfg["training"]["eval_every"] == 0:
            metrics = eval_musdb(model, vae, clap, noise_schedule, sampler, cfg, device)
            for k, v in metrics.items():
                writer.add_scalar(f"eval_sdr/{k}", v, step)
            print(f"[eval] epoch {epoch+1}: {metrics}")

    writer.close()
    save_checkpoint(
        str(Path(cfg["paths"]["checkpoint_dir"]) / "final.pt"),
        model, optimizer, scheduler, step, cfg["training"]["epochs"], cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["a", "b", "c"], default="b")
    parser.add_argument("--init_checkpoint", default=None)
    args = parser.parse_args()
    train(args)
