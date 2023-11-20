# -*- coding: utf-8 -*-
import logging
from typing import Tuple

from torch import randperm
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def split_edges_and_sample_negatives(data: Data, train_perc: float, valid_perc: float) -> Tuple[Data, Data, Data]:
    """
    Splits edges of the graph into training, validation, and test sets and performs negative sampling.

    Args:
    data (Data): A torch_geometric Data object containing the graph data, including edge indices.

    Returns:
    Tuple[Data, Data, Data]: A tuple containing Data objects for training, validation, and testing.
                              Each Data object includes features (x), edge indices (edge_index),
                              positive edges for training/validation/testing (train_pos_edge_index,
                              val_pos_edge_index, test_pos_edge_index) and negative edges
                              (train_neg_edge_index, val_neg_edge_index, test_neg_edge_index).
    """
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

    # Creating Data objects for training, validation, and testing
    train_data = Data(
        x=data.x, edge_index=train_edge, train_pos_edge_index=train_edge, train_neg_edge_index=train_edge_neg
    )
    val_data = Data(x=data.x, edge_index=val_edge, val_pos_edge_index=val_edge, val_neg_edge_index=val_edge_neg)
    test_data = Data(x=data.x, edge_index=test_edge, test_pos_edge_index=test_edge, test_neg_edge_index=test_edge_neg)

    return train_data, val_data, test_data
