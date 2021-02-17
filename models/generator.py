import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size=100, base_size=1024):
        super().__init__()

        self.latent_size = latent_size

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, base_size, 4, 1, padding=0, bias=False),
            nn.BatchNorm2d(base_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_size, base_size // 2, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_size // 2, base_size // 4, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_size // 4, base_size // 8, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size // 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_size // 8, 3, 4, 2, padding=1, bias=False),
            nn.Tanh()
        )

    def get_sample(self, batch_size, device):
        return torch.randn(batch_size, self.latent_size, 1, 1).to(device)

    def forward(self, x):
        x = self.conv_layers(x)
        return x