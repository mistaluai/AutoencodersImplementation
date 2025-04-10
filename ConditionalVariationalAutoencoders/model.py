import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_size=784, latent_size=50, num_classes=10):
        super(CVAE, self).__init__()
        self.latent = latent_size
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_size + num_classes, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU()
        )
        self.mu_fc = nn.Linear(200, latent_size)
        self.logvar_fc = nn.Linear(200, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, input_size)
        )

    def encode(self, x, c):
        c = F.one_hot(c, num_classes=self.num_classes)
        x = torch.cat((x, c), dim=-1)
        x = self.encoder(x)
        return self.mu_fc(x), self.logvar_fc(x)

    def decode(self, z, c):
        c = F.one_hot(c, num_classes=self.num_classes)
        z = torch.cat((z, c), dim=-1)
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar