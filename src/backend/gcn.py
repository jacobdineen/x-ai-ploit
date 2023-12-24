import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate

        # Construct hidden layers dynamically
        prev_dim = num_features
        for hidden_dim in hidden_dims:
            self.layers.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim

    def forward(self, x, edge_index):
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def decode(self, z, edge_index):
        # Compute edge scores as dot products of node embeddings
        edge_scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return edge_scores
