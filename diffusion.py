import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, timesteps, schedule, device):
        self.device = device

        self.timesteps = timesteps
        self.betas = self._linear_beta_schedule(self.timesteps)
        if schedule == "cosine":
            self.betas = self._cosine_beta_schedule(self.timesteps)
            print("using cosine schedule... \n")
        elif schedule == "sigmoid":
            self.betas = self._sigmoid_beta_schedule(self.timesteps)
            print("using sigmoid schedule... \n")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            # return mean

            img = mean
            # return torch.log(torch.clamp(img, min=1e-5) * 1)
            # return img.clamp(-6.0, 6.0)
            return img.clamp(-1.0, 1.0)
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # return mean + torch.sqrt(posterior_variance_t) * noise

            img = mean + torch.sqrt(posterior_variance_t) * noise
            # return torch.log(torch.clamp(img, min=1e-5) * 1)
            # return img.clamp(-6.0, 6.0)
            return img.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, model, cond, shape):
        device = next(model.parameters()).device
        batch_size = shape[0]
        images = []
        img = torch.randn(shape, device=device)
        # img = img.clamp_(0., 1.)
        images.append(img)
        for time in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((batch_size,), time, device=device, dtype=torch.long),
                time,
                cond,
            )
            images.append(img)
        return images

    def p_losses(self, denoise_model, x_start, t, cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predicted_noise = denoise_model(x_noisy, t)
        predicted_noise = denoise_model(x_noisy, t, cond)

        # L2 loss
        loss = F.mse_loss(noise, predicted_noise)

        return loss

    def _linear_beta_schedule(self, time_steps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, time_steps)

    def _cosine_beta_schedule(self, time_steps, s=0.008):
        t = torch.linspace(0, time_steps, time_steps + 1)
        alphas_cumprod = (
            torch.cos(((t / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    #     s_lim = 1.0
    #     t = torch.arange(timesteps)
    #     x = -s_lim + (2 * t * s_lim / timesteps)
    #     return beta_start + torch.sigmoid(x) * (beta_end - beta_start)

    def _sigmoid_beta_schedule(self, time_steps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, time_steps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def _extract(self, arr, t, x_start_shape):
        batch_size = t.shape[0]
        out = torch.gather(arr, dim=-1, index=t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_start_shape) - 1))).to(t.device)
