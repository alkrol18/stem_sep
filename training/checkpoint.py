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
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0), ckpt.get("config", {})
