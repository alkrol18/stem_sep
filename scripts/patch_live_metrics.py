#!/usr/bin/env python
"""Adds per-K-step validation loss + console prints to train.py.
Adds matching config knobs to dcc.yaml. Idempotent-ish (re-running may double-patch)."""
from pathlib import Path
import shutil
import re

train_path = Path("/home/users/ak724/stem_sep/training/train.py")
config_path = Path("/home/users/ak724/stem_sep/configs/dcc.yaml")

# Backup
shutil.copy(train_path, train_path.with_suffix(".py.bak"))
shutil.copy(config_path, config_path.with_suffix(".yaml.bak"))
print(f"Backups: {train_path}.bak, {config_path}.bak")

src = train_path.read_text()

# 1) imports
src = src.replace(
    "from torch.utils.data import DataLoader",
    "from torch.utils.data import DataLoader, Subset",
)

# 2) dataset/loader -> train+val split
old = '''    # Data
    index_path = str(Path(cfg["paths"]["moisesdb_root"]) / "index.json")
    dataset = MoisesDataset(index_path, cfg, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )'''
new = '''    # Data — train/val split
    index_path = str(Path(cfg["paths"]["moisesdb_root"]) / "index.json")
    train_dataset = MoisesDataset(index_path, cfg, augment=True)
    val_dataset = MoisesDataset(index_path, cfg, augment=False)
    n = len(train_dataset)
    val_frac = cfg["training"].get("val_frac", 0.1)
    n_val = max(1, int(n * val_frac))
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=g).tolist()
    dataset = Subset(train_dataset, indices[:n - n_val])
    val_subset = Subset(val_dataset, indices[n - n_val:])
    loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_iter = iter(val_loader)'''
assert old in src, "data block not found — train.py may have changed"
src = src.replace(old, new)

# 3) Subset doesn't have .stem_vocab — point to underlying dataset
src = src.replace("vocab = dataset.stem_vocab", "vocab = train_dataset.stem_vocab")

# 4) inject run_validation helper above train()
helper = '''
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
            mix_wav = batch["mixture_wav"].to(device)
            tgt_wav = batch["target_stem_wav"].to(device)
            queries = batch["query_text"]
            mix_mel = to_mel(mix_wav.squeeze(1).unsqueeze(1), cfg["audio"])
            tgt_mel = to_mel(tgt_wav.squeeze(1).unsqueeze(1), cfg["audio"])
            mix_latent = vae.encode(mix_mel)
            tgt_latent = vae.encode(tgt_mel)
            clap_emb = clap.get_text_embedding(list(queries))
            B = tgt_latent.shape[0]
            t = torch.randint(0, cfg["diffusion"]["num_timesteps"], (B,), device=device)
            x_t, noise = noise_schedule.q_sample(tgt_latent, t)
            pred_noise = model(x_t, mix_latent, clap_emb, t)
            loss = diffusion_mse_loss(pred_noise, noise)
            losses.append(loss.item())
    model.train()
    return (sum(losses) / max(1, len(losses))), val_iter


def train(args):'''
src = src.replace("def train(args):", helper)

# 5) inject prints + val into the loop, just before `step += 1`
inject = '''            if step % cfg["training"].get("print_every_steps", 50) == 0:
                print(f"[train] step={step} epoch={epoch} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            if step > 0 and step % cfg["training"].get("val_every_steps", 200) == 0:
                val_loss, val_iter = run_validation(
                    model, val_iter, val_loader, vae, clap, noise_schedule, cfg, device,
                    cfg["training"].get("val_batches", 4),
                )
                writer.add_scalar("loss/val", val_loss, step)
                print(f"[val] step={step} val_loss={val_loss:.4f}", flush=True)
            step += 1'''
src = src.replace("            step += 1", inject)

train_path.write_text(src)
print(f"Patched {train_path}")

# 6) config knobs
cfg_text = config_path.read_text()
extra = "  val_frac: 0.1\n  val_every_steps: 200\n  val_batches: 4\n  print_every_steps: 50\n"
if "val_every_steps" in cfg_text:
    print(f"{config_path} already patched, skipped")
else:
    pat = re.compile(r"(^training:\s*\n(?:[ \t]+.+\n)*)", re.MULTILINE)
    m = pat.search(cfg_text)
    if m:
        config_path.write_text(cfg_text.replace(m.group(1), m.group(1) + extra, 1))
        print(f"Patched {config_path}")
    else:
        print(f"WARN: 'training:' block not found — add these manually:\n{extra}")
