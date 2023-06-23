import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.in_channels = in_channels
        self.output_size = output_size

        self.decoder = nn.Sequential(nn.Linear(self.in_channels, self.in_channels//2),
                                nn.ReLU(),
                                nn.Linear(self.in_channels//2, self.in_channels//4),
                                nn.ReLU(),
                                nn.Linear(self.in_channels//4, self.output_size))

    def forward(self, x):
        return self.decoder(x)