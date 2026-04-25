"""Checkpoint save/load utilities."""
import os
from pathlib import Path
import torch


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    config: dict,
    ema=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step":            step,
        "epoch":           epoch,
        "config":          config,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, ema=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if ema is not None:
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        else:
            # Old checkpoint without EMA key: bootstrap shadow from loaded model weights.
            ema.reset_from_model(model)
            print("[checkpoint] no 'ema' key; EMA bootstrapped from loaded model weights")
    return ckpt.get("step", 0), ckpt.get("epoch", 0), ckpt.get("config", {})
