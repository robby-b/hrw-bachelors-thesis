import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_unit):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_unit = hidden_unit

        # define attention layer
        self.q_lin = nn.Linear(self.in_channels, self.hidden_unit)
        self.k_lin = nn.Linear(self.in_channels, self.hidden_unit)
        self.v_lin = nn.Linear(self.in_channels, self.hidden_unit)

    def forward(self, x):

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attention_weights, value)
