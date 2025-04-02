import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_size=784, latent_size=50):
        super(VAE, self).__init__()
        self.latent = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU()
        )
        self.mu_fc = nn.Linear(200, latent_size)
        self.logvar_fc = nn.Linear(200, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, input_size)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.mu_fc(x), self.logvar_fc(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar