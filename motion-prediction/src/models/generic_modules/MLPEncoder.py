import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.in_channels = in_channels
        self.output_size = output_size

        self.encoder = nn.Sequential(nn.Linear( self.in_channels, self.output_size//4),
                                     nn.ReLU(),
                                     nn.Linear(self.output_size//4, self.output_size//2),
                                     nn.ReLU(),
                                     nn.Linear(self.output_size//2, self.output_size),
                                     nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


