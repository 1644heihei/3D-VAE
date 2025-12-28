import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbeddings(nn.Module):
    """
    Time step embedding (Sinusoidal Positional Encoding)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Simple Residual Block for 1D vector
    """

    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.linear1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.linear2(h)

        return x + h


class DiffusionNetwork(nn.Module):
    """
    Denoising Network (Residual MLP)
    Input: Noisy Latent (z_t) + Condition Latent (z_lr) + Time (t)
    Output: Predicted Noise
    """

    def __init__(self, latent_dim=1024, hidden_dim=2048, num_layers=6):
        super().__init__()

        # Time Embedding
        time_dim = hidden_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input Projection
        # Input is concatenation of Noisy Latent and Condition Latent (LR)
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        # Residual Layers
        self.layers = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_dim) for _ in range(num_layers)]
        )

        # Output Projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, t, condition):
        """
        x: Noisy Latent (Batch, Latent_Dim)
        t: Timestep (Batch,)
        condition: LR Latent (Batch, Latent_Dim)
        """
        t_emb = self.time_mlp(t)

        # Concatenate input and condition
        h = torch.cat([x, condition], dim=1)
        h = self.input_proj(h)

        for layer in self.layers:
            h = layer(h, t_emb)

        return self.output_proj(h)


class DiffusionManager:
    """
    Manages noise schedule and sampling (DDPM/DDIM)
    """

    def __init__(
        self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Define beta schedule (Linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start, t):
        """
        Forward pass: q(x_t | x_0)
        Returns: x_t, noise
        """
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    @torch.no_grad()
    def sample_ddim(self, model, condition, num_inference_steps=50, eta=0.0):
        """
        Fast sampling using DDIM
        """
        batch_size = condition.shape[0]
        latent_dim = condition.shape[1]

        # Start from random noise
        x = torch.randn(batch_size, latent_dim).to(self.device)

        # Create time steps for DDIM (e.g., 0, 20, 40, ..., 980)
        times = (
            torch.linspace(0, self.num_timesteps - 1, num_inference_steps)
            .long()
            .to(self.device)
        )
        times = list(reversed(times))  # 980, ..., 0

        for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
            t_tensor = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )

            # Predict noise
            predicted_noise = model(x, t_tensor, condition)

            # DDIM Step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = (
                self.alphas_cumprod[times[i + 1]]
                if i < len(times) - 1
                else torch.tensor(1.0).to(self.device)
            )

            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(
                alpha_t
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise

            # Noise
            noise = sigma_t * torch.randn_like(x)

            # Update x
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + noise

        return x
