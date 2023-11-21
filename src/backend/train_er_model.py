# -*- coding: utf-8 -*-
"""
Module to train a graph convolutional network (GCN) model on a given graph dataset.

"""
import argparse
import logging
from typing import Any, Tuple

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from src.backend.gcn import GCN
from src.backend.generate_er_graphs import CVEGraphGenerator
from src.backend.utils.graph_utils import split_edges_and_sample_negatives
from src.backend.utils.modeling_utils import compute_metrics

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, criterion: Any, device: torch.device
) -> Tuple[float, dict]:
    """
    Train the model for one epoch.

    Args:
        model (Module): The graph convolutional network (GCN) model to be trained.
        data (Data): The data object from torch_geometric containing graph data including node features and edge indices.
        optimizer (Optimizer): The optimizer to be used for training.
        criterion (Any): The loss function used for training.
        device (torch.device): The device (CPU or CUDA) on which the model is being trained.

    Returns:
        Tuple[float, dict]: A tuple containing the loss value for the epoch (as a float) and a dictionary of computed metrics.
    """
    model.train()
    optimizer.zero_grad()

    z = model(data.x.to(device), data.edge_index.to(device))
    logits = model.decode(z, data.train_pos_edge_index.to(device), data.train_neg_edge_index.to(device))
    labels = torch.cat(
        [torch.ones(data.train_pos_edge_index.size(1)), torch.zeros(data.train_neg_edge_index.size(1))], dim=0
    ).to(device)

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    predictions = torch.sigmoid(logits) > 0.5
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    metrics = compute_metrics(labels, predictions, logits)
    return loss.item(), metrics


def eval_epoch(model: torch.nn.Module, data: Data, criterion: Any, device: torch.device) -> Tuple[float, dict]:
    """
    Evaluate the model on validation or test data for one epoch.

    Args:
        model (Module): The graph convolutional network (GCN) model to be evaluated.
        data (Data): The data object from torch_geometric containing graph data including node features and edge indices for validation or testing.
        criterion (Any): The loss function used for evaluation.
        device (torch.device): The device (CPU or CUDA) on which the model is being evaluated.

    Returns:
        Tuple[float, dict]: A tuple containing the loss value for the epoch (as a float) and a dictionary of computed metrics.
    """
    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.edge_index.to(device))
        logits = model.decode(z, data.val_pos_edge_index.to(device), data.val_neg_edge_index.to(device))
        labels = torch.cat(
            [torch.ones(data.val_pos_edge_index.size(1)), torch.zeros(data.val_neg_edge_index.size(1))], dim=0
        ).to(device)

        loss = criterion(logits, labels)

        predictions = torch.sigmoid(logits) > 0.5
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        metrics = compute_metrics(labels, predictions, logits)
        return loss.item(), metrics


def prepare_data(graph_save_path: str, features_path: str, vectorizer_path: str) -> Tuple[Any, int]:
    """
    Prepares graph data for the GCN model from a given file path.

    Args:
        graph_save_path (str): The path to the saved graph.
        features_path (str): The path to the saved features.
        vectorizer_path (str): The path to the saved vectorizer.


    Returns:
        Tuple[Any, int]: A tuple containing the prepared data and the number of features.
                        The prepared data is a torch_geometric Data object with node features and edge indices.
                        The number of features is an integer representing the size of the feature vector for each node.
    """
    logging.info("Loading graph data...")
    generator = CVEGraphGenerator(file_path="")
    generator.load_graph(graph_save_path, features_path, vectorizer_path)
    graph = generator.graph
    num_features = len(generator.vectorizer.get_feature_names_out())

    default_vector = [0] * num_features  # Replace with appropriate length
    logging.info('Adding "vector" attribute to nodes...')
    for node in tqdm(graph.nodes):
        if "vector" not in graph.nodes[node]:
            graph.nodes[node]["vector"] = default_vector

    data = from_networkx(graph)
    logging.info("nx graph transformed to torch_geometric data object")
    node_features = [graph.nodes[node]["vector"] for node in graph.nodes()]
    data.x = torch.tensor(node_features, dtype=torch.float)
    return data, num_features


def main(
    graph_save_path: str,
    features_path: str,
    vectorizer_path: str,
    train_percent: float = 0.80,
    valid_percent: float = 0.10,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    hidden_dim: int = 64,
    dropout_rate: float = 0.5,
):
    """
    main logic to grab data, train model, and plot results

    Args:
        graph_path (str): The path to the saved graph.
        train_percent (float): The percentage of data to use for training.
        valid_percent (float): The percentage of data to use for validation.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        hidden_dim (int): The number of hidden dimensions.
        dropout_rate (float): The dropout rate for the model.

    Returns:
        None
    """
    logging.info("Training GCN model...")
    # Prepare data
    data, num_features = prepare_data(graph_save_path, features_path, vectorizer_path)
    train_data, val_data, _ = split_edges_and_sample_negatives(data, train_perc=train_percent, valid_perc=valid_percent)

    model = GCN(num_features=num_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    logging.info(f"model loaded onto device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_keys = ["loss", "accuracy", "precision", "recall", "f1", "aucroc"]
    metrics = {phase: {key: [] for key in metric_keys} for phase in ["train", "val"]}

    for epoch in tqdm(range(num_epochs)):
        train_metrics = train_epoch(model, train_data, optimizer, criterion, device)
        val_metrics = eval_epoch(model, val_data, criterion, device)

        epoch_metrics = {
            "train": train_metrics,  # Directly use the tuple
            "val": val_metrics,  # Directly use the tuple
        }

        for phase in ["train", "val"]:
            for key, value in zip(metric_keys, epoch_metrics[phase]):
                metrics[phase][key].append(value)

        train_metrics_formatted = " - ".join(f"{metric:.4f}" for metric in train_metrics[1])
        val_metrics_formatted = " - ".join(f"{metric:.4f}" for metric in val_metrics[1])
        logging.info(f"Epoch {epoch+1}:")
        logging.info(f"Train Loss: {train_metrics[0]:.4f} - Val Loss: {val_metrics[0]:.4f}")
        logging.info(f"Train Metrics: {train_metrics_formatted}")
        logging.info(f"Val Metrics: {val_metrics_formatted}")

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train"]["loss"], label="Training Loss")
    plt.plot(metrics["val"]["loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="GCN Training Script")
    parser.add_argument("--graph_save_path", type=str, default="data/graph.gml", help="Path to the nx graph to")
    parser.add_argument("--feature_save_path", type=str, default="data/features.npz", help="Path to the nx graph to")
    parser.add_argument(
        "--vectorizer_save_path", type=str, default="data/vectorizer.pkl", help="Path to the nx graph to"
    )
    parser.add_argument("--train_perc", type=float, default=0.80, help="Percent of data to use for training")
    parser.add_argument("--valid_perc", type=float, default=0.10, help="Percent of data to use for validation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Number of hidden dimensions")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the model")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.graph_save_path,
        args.feature_save_path,
        args.vectorizer_save_path,
        args.train_perc,
        args.valid_perc,
        args.num_epochs,
        args.learning_rate,
        args.weight_decay,
        args.hidden_dim,
        args.dropout_rate,
    )
