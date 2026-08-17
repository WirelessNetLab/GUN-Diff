import torch
import numpy as np

def beta_cosine(timesteps, s=0.008, dtype=torch.float32):

    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def beta_linear(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

def beta_vp(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

def get_beta_schedule(mode, beta_min, beta_max, timesteps, **kwargs):
    if mode == 'cosine':
        return beta_cosine(timesteps, s=0.008, dtype=torch.float32)
    elif mode == 'lin':
        return  beta_linear(timesteps, beta_start=beta_min, beta_end=beta_max, dtype=torch.float32)
    elif mode == 'vp':
        return beta_vp(timesteps, dtype=torch.float32)
    else:
        raise ValueError(f"Scale error, there is no {mode} type increment, please use: cosine, lin, vp")
