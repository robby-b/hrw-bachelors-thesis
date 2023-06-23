import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class SubGraphLayer(nn.Module):
    def __init__(self, in_channels, hidden_unit):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_unit = hidden_unit

        # single mlp layer
        self.encoder = nn.Sequential(   nn.Linear(self.in_channels, self.hidden_unit),
                                        nn.LayerNorm(self.hidden_unit),
                                        nn.ReLU())

    def forward(self, x, cluster):

        # encode input
        x = self.encoder(x)

        # max pool encoded input
        x_aggr = scatter(x, cluster, dim=1, reduce='max')
        # expand along index/cluster to same shape as encoded x with max elements for every polyline at right index
        index_mask = cluster.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x_aggr = x_aggr.gather(dim=1, index=index_mask)

        # concat max pooled vectors with encoded vectors
        return torch.cat([x, x_aggr], dim=2)


# Subgraph
class SubGraph(nn.Module):
    def __init__(self, in_channels, num_subgraph_layers, hidden_unit):
        super().__init__()

        self.hidden_unit = hidden_unit
        self.in_channels = in_channels
        self.num_subgraph_layers = num_subgraph_layers

        # stack subgraph layer
        self.subgraph_layers = nn.ModuleList()
        for i in range(self.num_subgraph_layers):
            self.subgraph_layers.append(SubGraphLayer(self.in_channels, self.hidden_unit))
            self.in_channels = self.hidden_unit*2 # output size after first pass is number of hidden_units x 2

    def forward(self, x, cluster):

        # pass data through subgraph layers
        for layer in self.subgraph_layers:
            x = layer(x, cluster)

        # pool data
        x = scatter(x, cluster, dim=1, reduce='max')
        return F.normalize(x)
