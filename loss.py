import torch
import torch.nn.functional as F
from sample import q_sample


def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    # L2 loss
    loss = F.mse_loss(noise, predicted_noise)

    return loss