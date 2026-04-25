"""Noise schedule, forward process, DDIM sampler, and prediction utilities."""
import torch
import torch.nn as nn
import numpy as np


class NoiseSchedule:
    def __init__(self, cfg: dict):
        T = cfg["num_timesteps"]
        schedule = cfg["schedule"]

        if schedule == "linear":
            betas = np.linspace(cfg["beta_start"], cfg["beta_end"], T, dtype=np.float64)
        elif schedule == "cosine":
            # OpenAI cosine schedule
            steps = np.arange(T + 1) / T
            f = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
            betas = np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)

        self.T = T
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alpha_bars = torch.tensor(alpha_bars, dtype=torch.float32)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: x_t = sqrt(ā_t) * x0 + sqrt(1-ā_t) * ε
        Returns (x_t, noise).
        Works for any x0 dimensionality (3-D latents or 4-D mel latents).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # Reshape (B,) → (B, 1, 1, ...) to broadcast over all non-batch dims.
        view = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        sqrt_ab   = self.sqrt_alpha_bars[t].view(*view)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(*view)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor
    ) -> torch.Tensor:
        view      = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
        sqrt_ab   = self.sqrt_alpha_bars[t].view(*view)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(*view)
        return (x_t - sqrt_1mab * pred_noise) / sqrt_ab.clamp(min=1e-8)


class DDIMSampler:
    def __init__(self, schedule: NoiseSchedule, inference_steps: int = 50):
        self.schedule = schedule
        self.steps = inference_steps
        T = schedule.T
        # Evenly spaced timestep sequence
        self.timesteps = torch.linspace(T - 1, 0, inference_steps + 1).long()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        mixture_latent: torch.Tensor,
        clap_emb: torch.Tensor,
        shape: tuple,
        device: torch.device,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM deterministic denoising (eta=0).

        Args:
            model:           StemSeparationUNet
            mixture_latent:  (B, 4, F, T)
            clap_emb:        (B, 512)
            shape:           (B, 4, F, T) — must match mixture_latent
            device:
            eta:             stochasticity (0 = deterministic)
        Returns:
            x0: (B, 4, F, T) predicted clean stem latent
        """
        x = torch.randn(shape, device=device)
        sched = self.schedule

        for i in range(len(self.timesteps) - 1):
            t_cur = self.timesteps[i]
            t_next = self.timesteps[i + 1]

            t_batch = torch.full((shape[0],), t_cur.item(), device=device, dtype=torch.long)
            pred_noise = model(x, mixture_latent, clap_emb, t_batch)

            ab_cur = sched.alpha_bars[t_cur].to(device)
            ab_next = sched.alpha_bars[t_next].to(device)

            x0_pred = (x - (1 - ab_cur).sqrt() * pred_noise) / ab_cur.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-1, 1)

            if t_next == 0:
                x = x0_pred
            else:
                dir_xt = (1 - ab_next).sqrt() * pred_noise
                x = ab_next.sqrt() * x0_pred + dir_xt

        return x
