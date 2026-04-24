"""Loss functions: diffusion MSE, smoothness, consistency."""
import random
import torch
import torch.nn.functional as F


def diffusion_mse_loss(pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise, true_noise)


def smoothness_loss(predicted_stem_latent: torch.Tensor) -> torch.Tensor:
    """Penalise large frame-to-frame differences along the time axis."""
    diff = predicted_stem_latent[..., 1:] - predicted_stem_latent[..., :-1]
    return F.mse_loss(diff, torch.zeros_like(diff))


def consistency_loss(
    model,
    mix_latent: torch.Tensor,
    stem_queries: list[tuple[torch.Tensor, str]],
    noise_schedule,
    t_batch: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """
    Encourages that the sum of predicted clean stems equals the mixture latent.

    Args:
        model:          StemSeparationUNet
        mix_latent:     (B, 4, F, T) clean mixture latent
        stem_queries:   list of (clap_emb (B, 512), stem_name) — up to stem_cap
        noise_schedule: NoiseSchedule
        t_batch:        (B,) current timesteps (only called when t mean < max_t)
        cfg:            losses sub-dict
    Returns:
        scalar loss tensor
    """
    cap = cfg.get("consistency_stem_cap", 4)
    if len(stem_queries) > cap:
        stem_queries = random.sample(stem_queries, cap)

    pred_sum = None
    for clap_emb, _ in stem_queries:
        # Sample fresh noise for this stem
        noise = torch.randn_like(mix_latent)
        x_t, _ = noise_schedule.q_sample(mix_latent, t_batch, noise)
        pred_noise = model(x_t, mix_latent, clap_emb, t_batch)
        x0_pred = noise_schedule.predict_x0_from_noise(x_t, t_batch, pred_noise)
        pred_sum = x0_pred if pred_sum is None else pred_sum + x0_pred

    if pred_sum is None:
        return torch.tensor(0.0, device=mix_latent.device)

    return F.mse_loss(pred_sum, mix_latent)
