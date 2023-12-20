# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        prev_dim = num_features

        for hidden_dim in hidden_dims:
            self.layers.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # No activation and dropout on the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
