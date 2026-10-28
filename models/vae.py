import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: (3, 64, 64) -> (256, 16, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.GroupNorm(8, 256), nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(256, latent_channels, 1)
        self.to_logvar = nn.Conv2d(256, latent_channels, 1)

        # Decoder: (4, 16, 16) -> (3, 64, 64)
        self.decoder_input = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, 1, 1),
            nn.GroupNorm(8, 256), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-10, 10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar
