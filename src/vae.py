
import torch
import torch.nn as nn

Softplus = torch.nn.Softplus()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size

    def forward(self, input):
        return input.view(input.size(0), self.channel_size, 2, 1)

class VAE(nn.Module):
    """ Based on sksq96/pytorch-vae """
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2),
            nn.Softplus(),
            nn.Conv2d(16, 16, kernel_size=2, stride=2),
            nn.Softplus(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4),
            nn.Softplus(),
            nn.Conv2d(32, 32, kernel_size=2, stride=2),
            nn.Softplus(),
            Flatten()
        )

        self.mu = nn.Linear(self.encoder[-3].out_channels * 2, latent_dim)
        self.log_sd = nn.Linear(self.encoder[-3].out_channels * 2, latent_dim)
        self.log_obs_sd = torch.nn.Parameter(torch.ones(1))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder[-3].out_channels * 2),
            UnFlatten(channel_size=self.encoder[-3].out_channels),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        z_mu, z_log_sd = self.mu(h), self.log_sd(h)
        z_dist = torch.distributions.Normal(z_mu, Softplus(z_log_sd))
        z = z_dist.rsample()
        return z_dist, self.decoder(z)
