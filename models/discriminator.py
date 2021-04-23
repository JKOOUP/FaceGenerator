import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, base_size=64):
        super().__init__()

        self.base_size = base_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, base_size, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(base_size, base_size * 2, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size * 2),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(base_size * 2, base_size * 4, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size * 4),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(base_size * 4, base_size * 8, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base_size * 8),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(base_size * 8, 2, 4, 1, padding=0, bias=False),
        )
     
    def forward(self, x):
        x = self.conv_layers(x)
        return x