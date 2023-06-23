import torch.nn as nn


class TrajDecoder(nn.Module):
    def __init__(self, in_channels, hidden_unit, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_unit = hidden_unit
        self.out_channels = out_channels

        # decoder is simple mlp
        self.decoder = nn.Sequential(   nn.Linear(in_channels, hidden_unit),
                                        nn.LayerNorm(hidden_unit),
                                        nn.ReLU(),
                                        nn.Linear(hidden_unit, out_channels))
                                        
    def forward(self, x):
        return self.decoder(x)

