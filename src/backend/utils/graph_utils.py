# -*- coding: utf-8 -*-
import logging
from typing import Tuple

# from torch_geometric.loader import DataLoader
# import torch
from torch import randperm
from torch_geometric.data import Data

# from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

# https://chat.openai.com/share/e9495e15-6218-4cfa-b4e2-b10466a807ff
#  need to have custom dataloaders here for batch training


def split_edges_and_sample_negatives(data: Data, train_perc: float, valid_perc: float) -> Tuple[Data, Data, Data]:
    num_edges = data.edge_index.size(1)
    num_nodes = data.num_nodes

    # Shuffle and split edges
    perm = randperm(num_edges)
    num_train = int(num_edges * train_perc)  # 80% for training
    num_val = int(num_edges * valid_perc)  # 10% for validation

    train_edge = data.edge_index[:, perm[:num_train]]
    val_edge = data.edge_index[:, perm[num_train : num_train + num_val]]
    test_edge = data.edge_index[:, perm[num_train + num_val :]]

    # Negative sampling
    train_edge_neg = negative_sampling(
        edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=train_edge.size(1)
    )
    val_edge_neg = negative_sampling(edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=val_edge.size(1))
    test_edge_neg = negative_sampling(
        edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=test_edge.size(1)
    )

    train_pos_data = Data(x=data.x, edge_index=train_edge)
    train_neg_data = Data(x=data.x, edge_index=train_edge_neg)
    val_pos_data = Data(x=data.x, edge_index=val_edge)
    val_neg_data = Data(x=data.x, edge_index=val_edge_neg)
    test_pos_data = Data(x=data.x, edge_index=test_edge)
    test_neg_data = Data(x=data.x, edge_index=test_edge_neg)
    return train_pos_data, train_neg_data, val_pos_data, val_neg_data, test_pos_data, test_neg_data


def create_edge_batches(edge_index, batch_size, num_nodes, node_features):
    total_edges = edge_index.size(1)
    for start in range(0, total_edges, batch_size):
        end = min(start + batch_size, total_edges)
        batch_edges = edge_index[:, start:end]
        print(f"Creating batch with {batch_edges.size(1)} edges")  # Debugging info
        yield Data(x=node_features, edge_index=batch_edges, num_nodes=num_nodes)


# if __name__ == "__main__":
#     from src.backend.generate_er_graphs import CVEGraphGenerator

#     logging.info("Loading graph data...")
#     generator = CVEGraphGenerator(file_path="")
#     graph_save_path = "data/samplesize_102/graph.gml"
#     vectorizer_path = "data/samplesize_102/ft_model.bin"
#     generator.load_graph(graph_save_path, vectorizer_path)
#     graph = generator.graph
#     print("graph num nodes: ", graph.number_of_nodes())
#     print("graph num edges: ", graph.number_of_edges())
#     num_features = generator.ft_model.get_dimension()
#     logging.info("Number of features: %d", num_features)

#     data = from_networkx(graph)
#     logging.info("nx graph transformed to torch_geometric data object")
#     node_features = [graph.nodes[node]["vector"] for node in graph.nodes()]

#     data.x = torch.tensor(node_features, dtype=torch.float)
#     train_percent = 0.8
#     valid_percent = 0.1
#     batch_size = 32  # Define your batch size
# train_pos_data, train_neg_data, val_pos_data, val_neg_data = split_edges_and_sample_negatives(data, train_percent, valid_percent)
# train_pos_batches = create_edge_batches(train_pos_data.edge_index, batch_size, data.num_nodes, data.x)
# train_neg_batches = create_edge_batches(train_neg_data.edge_index, batch_size, data.num_nodes, data.x)
# (
#     train_pos_batches,
#     train_neg_batches,
#     val_pos_batches,
#     val_neg_batches,
#     test_pos_batches,
#     test_neg_batches,
# ) = split_sample_yield(data, train_percent, valid_percent, batch_size)
# for batch in train_pos_batches:
#     # Process your batch here
#     # For instance:
#     print(f"Batch size (edges): {batch.edge_index.size(1)}")
#     print(f"Number of nodes (x): {batch.x.size(0)}")

# train_pos_loader = DataLoader(train_pos_data, batch_size=batch_size, follow_batch=['x', 'edge_index'])
# train_neg_loader = DataLoader(train_neg_data, batch_size=batch_size, follow_batch=['x', 'edge_index'])
# val_pos_loader = DataLoader(val_pos_data, batch_size=batch_size, follow_batch=['x', 'edge_index'])
# val_neg_loader = DataLoader(val_neg_data, batch_size=batch_size, follow_batch=['x', 'edge_index'])
# for batch_idx, batch in enumerate(train_pos_loader):
#     print(f"Batch {batch_idx}:")
#     print(f" - Number of nodes (x): {batch.x.size(0)}")
#     print(f" - Number of edges: {batch.edge_index.size(1)}")
#     if hasattr(batch, 'batch'):
#         print(f" - Batch attribute size: {batch.batch.size(0)}")
