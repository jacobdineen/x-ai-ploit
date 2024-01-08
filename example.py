import torch
import torch_geometric.transforms as T
from joblib import load
from sklearn.metrics import roc_auc_score

# from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling
from tqdm import tqdm

# from torch_geometric.transforms import RandomLinkSplit
# import os.path as osp


if torch.cuda.is_available():
    device = torch.device("cpu")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

graph_path = "/home/jdineen/Documents/xaiploit/data/samplesize_105/graph.gml"
nx_graph = load(graph_path)
data = from_networkx(nx_graph)
print(data)

transform = T.Compose(
    [
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.5, is_undirected=True, add_negative_train_samples=False),
    ]
)
train_data, val_data, test_data = transform(data)
# Print the number of nodes in each dataset
print(f"Number of nodes in full data: {data.num_nodes}")
print(f"Number of nodes in train data: {train_data.num_nodes}")
print(f"Number of nodes in validation data: {val_data.num_nodes}")
print(f"Number of nodes in test data: {test_data.num_nodes}")

# Print the number of edges in each dataset
# For link prediction, you'll typically look at edge_label_index to understand the number of edges
print(f"Number of edges in full data: {data.edge_index.size(1) // 2}")  # Dividing by 2 because it's undirected
print(f"Number of edges in train data: {train_data.edge_label_index.size(1)}")
print(f"Number of edges in validation data: {val_data.edge_label_index.size(1)}")
print(f"Number of edges in test data: {test_data.edge_label_index.size(1)}")


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(300, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.vector, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([train_data.edge_label, train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    # get the predicted label

    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.vector, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    torch.cuda.empty_cache()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 2):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, " f"Test: {test_auc:.4f}")

print(f"Final Test: {final_test_auc:.4f}")


def decode_all_node_by_node(z, threshold=0.5):
    num_nodes = z.size(0)
    final_edge_index = []

    for i in tqdm(range(num_nodes)):
        # Compute scores between node 'i' and all other nodes
        scores = torch.matmul(z[i].unsqueeze(0), z.t()).squeeze(0)

        # Find indices where the score exceeds the threshold, avoiding self-loop
        indices = torch.where(scores > threshold)[0]
        indices = indices[indices != i]

        # Create edge pairs and add to final list
        edges = torch.stack([torch.full_like(indices, i, dtype=torch.long), indices], dim=0)
        final_edge_index.append(edges)

    # Concatenate all edge indices
    final_edge_index = torch.cat(final_edge_index, dim=1)

    return final_edge_index.to(z.device)


# Encoding step
z = model.encode(test_data.vector, test_data.edge_index)

# Decoding step with batching
final_edge_index = decode_all_node_by_node(z)
print(final_edge_index.size())
