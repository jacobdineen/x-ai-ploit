# -*- coding: utf-8 -*-
"""
Module to train a graph convolutional network (GCN) model on a given graph dataset.

"""
import argparse
import logging
import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from src.backend.gcn import GCN
from src.backend.generate_er_graphs import CVEGraphGenerator
from src.backend.utils.graph_utils import create_edge_batches, split_edges_and_sample_negatives

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")
# Seed this script for reproducibility


def compute_metrics(labels, predictions, loss):
    labels = torch.clamp(torch.round(labels), 0, 1).bool()
    preds = predictions.bool()
    # True Positives, False Positives, and False Negatives
    tp = torch.sum(preds & labels).item()
    fp = torch.sum(preds & ~labels).item()
    fn = torch.sum(~preds & labels).item()

    accuracy = torch.sum(preds == labels).item() / labels.numel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return loss, accuracy, precision, recall, f1


def train_epoch(
    model: torch.nn.Module,
    data: torch_geometric.data.Data,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    device: torch.device,
    pos_data: torch_geometric.data.Data,
    neg_data: torch_geometric.data.Data,
    batch_size: int = 32,
) -> Tuple[float, Tuple[float, float, float, float, float, str], torch.Tensor, np.ndarray, np.ndarray]:
    model.train()
    total_loss = 0
    batch_count = 0
    all_logits = []
    all_predictions = []
    all_labels = []

    # Need to recreate batches on every run?
    pos_batches = create_edge_batches(pos_data.edge_index, batch_size, data.num_nodes, data.x)
    neg_batches = create_edge_batches(neg_data.edge_index, batch_size, data.num_nodes, data.x)

    logging.info("starting epoch")
    for pos_batch, neg_batch in zip(pos_batches, neg_batches):
        optimizer.zero_grad(set_to_none=True)

        # Combine positive and negative edges and their respective node features
        edge_index = torch.cat([pos_batch.edge_index, neg_batch.edge_index], dim=1)
        x = data.x.to(device)

        # Forward pass
        z = model(x, edge_index.to(device))
        pos_edge_index = pos_batch.edge_index.to(device)
        neg_edge_index = neg_batch.edge_index.to(device)
        logits = model.decode(z, pos_edge_index, neg_edge_index)

        # Create labels for positive and negative edges
        labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

        # Loss computation and backpropagation
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        all_logits.append(logits)
        all_labels.append(labels)
        predictions = torch.sigmoid(logits) > 0.5
        all_predictions.append(predictions)
        batch_count += 1
        torch.cuda.empty_cache()

    # Concatenate all predictions and labels across batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / batch_count
    metrics = compute_metrics(all_labels, all_predictions, avg_loss)

    return avg_loss, metrics, all_logits, all_predictions.cpu().numpy(), all_labels.cpu().numpy()


def eval_epoch(
    model: torch.nn.Module,
    data: Data,
    criterion: Any,
    device: torch.device,
    pos_data: torch_geometric.data.Data,
    neg_data: torch_geometric.data.Data,
    batch_size: int = 32,
) -> Tuple[float, Tuple[float, float, float, float, float, str], torch.Tensor, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_logits = []
    all_labels = []
    all_predictions = []

    pos_batches = create_edge_batches(pos_data.edge_index, batch_size, data.num_nodes, data.x)
    neg_batches = create_edge_batches(neg_data.edge_index, batch_size, data.num_nodes, data.x)

    with torch.no_grad():
        for pos_batch, neg_batch in zip(pos_batches, neg_batches):
            if pos_batch.edge_index.size(1) == 0 or neg_batch.edge_index.size(1) == 0:
                logging.info("Skipping batch with no edges")
                continue
            edge_index = torch.cat([pos_batch.edge_index, neg_batch.edge_index], dim=1)
            x = data.x.to(device)

            # Forward pass
            z = model(x, edge_index.to(device))
            pos_edge_index = pos_batch.edge_index.to(device)
            neg_edge_index = neg_batch.edge_index.to(device)
            logits = model.decode(z, pos_edge_index, neg_edge_index)

            # Labels
            labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(
                device
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits)
            all_labels.append(labels)
            predictions = torch.sigmoid(logits) > 0.5
            all_predictions.append(predictions)
            batch_count += 1
            torch.cuda.empty_cache()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    avg_loss = total_loss / batch_count
    metrics = compute_metrics(all_labels, all_predictions, avg_loss)

    return avg_loss, metrics, all_logits, all_predictions.cpu().numpy(), all_labels.cpu().numpy()


def prepare_data(graph_save_path: str, vectorizer_path: str) -> Tuple[Any, int]:
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
    generator.load_graph(graph_save_path, vectorizer_path)
    graph = generator.graph
    num_features = generator.ft_model.get_dimension()
    logging.info("Number of features: %d", num_features)

    data = from_networkx(graph)
    logging.info("nx graph transformed to torch_geometric data object")
    node_features = [graph.nodes[node]["vector"] for node in graph.nodes()]

    data.x = torch.tensor(node_features, dtype=torch.float)

    return data, num_features


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save the training model at the checkpoint.

    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        filename (str): Name of the checkpoint file.
    """
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint into.

    Returns:
        int: The epoch to resume training from.
        float: The best validation loss recorded up to this checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_loss", float("inf"))


def _plot_results(metrics):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train"]["loss"], label="Training Loss")
    plt.plot(metrics["val"]["loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()


def main(
    read_dir: str,
    data_size: int,
    hidden_dims: list,
    checkpoint_path: str = "models/checkpoint.pth.tar",
    load_from_checkpoint: bool = False,
    **kwargs,
):
    """
    main logic to grab data, train model, and plot results

    Args:
        read_dir (str): The directory containing the files to read.
        hidden_dims (list): The list of hidden dimensions for each layer.
        checkpoint_path (str): The path to save the model checkpoint.

    kwargs:
        train_percent (float): The percentage of data to use for training.
        valid_percent (float): The percentage of data to use for validation.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        dropout_rate (float): The dropout rate for the model.
        plot_results (bool): Whether to plot the training and validation loss over epochs.
        batch_size (int): The batch size for training and validation.
        logging_interval (int): The interval at which to log metrics.
        load_from_checkpoint (bool): Whether to load the best model checkpoint.

    Returns:
        None
    """
    train_percent = kwargs.get("train_percent", 0.80)
    valid_percent = kwargs.get("valid_percent", 0.20)
    num_epochs = kwargs.get("num_epochs", 100)
    learning_rate = kwargs.get("learning_rate", 0.01)
    weight_decay = kwargs.get("weight_decay", 1e-5)
    dropout_rate = kwargs.get("dropout_rate", 0.5)
    plot_results = kwargs.get("plot_results", True)
    batch_size = kwargs.get("batch_size", 256)
    logging_interval = kwargs.get("logging_interval", 100)
    load_from_checkpoint = kwargs.get("load_from_checkpoint", False)

    # Seed here
    torch.manual_seed(0)
    np.random.seed(0)

    num_features = 300  # hard coded for now
    model = GCN(num_features=num_features, hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    logging.info(f"model loaded onto device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    if load_from_checkpoint:
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        logging.info(f"Loaded checkpoint at epoch {start_epoch} with best validation loss of {best_val_loss}")
        return None

    logging.info("Training GCN model...")

    # Prepare data
    # Construct paths dynamically based on the limit and output directory
    base_path = os.path.join(read_dir, f"samplesize_{data_size}" if data_size else "all_data")
    graph_save_path = os.path.join(base_path, "graph.gml")
    vectorizer_path = os.path.join(base_path, "ft_model.bin")

    data, num_features = prepare_data(graph_save_path, vectorizer_path)
    train_pos_data, train_neg_data, val_pos_data, val_neg_data, _, _ = split_edges_and_sample_negatives(
        data, train_percent, valid_percent
    )

    metric_keys = ["loss", "accuracy", "precision", "recall", "f1"]
    metrics = {phase: {key: [] for key in metric_keys} for phase in ["train", "val"]}

    best_val_loss = float("inf")  # Initialize best validation loss for checkpointing
    for epoch in tqdm(range(num_epochs)):
        try:
            torch.cuda.synchronize(device)
            train_metrics = train_epoch(
                model, data, optimizer, criterion, device, train_pos_data, train_neg_data, batch_size
            )
            val_metrics = eval_epoch(model, data, criterion, device, val_pos_data, val_neg_data, batch_size)

            epoch_metrics = {
                "train": train_metrics,  # Directly use the tuple
                "val": val_metrics,  # Directly use the tuple
            }

            for phase in ["train", "val"]:
                for key, value in zip(metric_keys, epoch_metrics[phase]):
                    metrics[phase][key].append(value)

            # Update the formatting to include metric names
            train_metrics_formatted = " - ".join(
                f"{name}: {metric:.4f}" for name, metric in zip(metric_keys, train_metrics[1])
            )
            val_metrics_formatted = " - ".join(
                f"{name}: {metric:.4f}" for name, metric in zip(metric_keys, val_metrics[1])
            )
            if epoch % logging_interval == 0:
                logging.info(f"Epoch {epoch+1}:")
                logging.info(f"Train Loss: {train_metrics[0]:.4f} - Val Loss: {val_metrics[0]:.4f}")
                logging.info(f"Train Metrics: {train_metrics_formatted}")
                logging.info(f"Val Metrics: {val_metrics_formatted}")

            # Checkpointing logic
            is_best = val_metrics[0] < best_val_loss
            if is_best:
                best_val_loss = val_metrics[0]
                logging.info(
                    "checkpointing model at epoch %d with best validation loss of %.3f", epoch + 1, best_val_loss
                )

                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_val_loss": best_val_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=checkpoint_path,
                )

        except RuntimeError as e:
            torch.cuda.synchronize(device)
            print(f"Runtime error during epoch {epoch+1}: {e}")
            break

    if plot_results:
        _plot_results(metrics)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="GCN Training Script")
    parser.add_argument("--read_dir", type=str, default="data", help="Path to the nx graph to")
    parser.add_argument("--data_size", type=int, default=None, help="Number of samples to use for training")
    parser.add_argument("--train_perc", type=float, default=0.70, help="Percent of data to use for training")
    parser.add_argument("--valid_perc", type=float, default=0.20, help="Percent of data to use for validation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help="List of hidden dimensions for each layer (default: 3 layers with 128 units each)",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("--logging_interval", type=int, default=100, help="logging interval for metrics")
    parser.add_argument(
        "--checkpoint_path", type=str, default="models/checkpoint.pth.tar", help="GCN model checkpoint path"
    )
    parser.add_argument("--load_from_checkpoint", type=bool, default=False, help="load best model checkpoint")
    parser.add_argument("--plot_results", type=bool, default=True, help="plot training and validation loss")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training and validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.read_dir,
        args.data_size,
        args.hidden_dims,
        checkpoint_path=args.checkpoint_path,
        load_from_checkpoint=args.load_from_checkpoint,
        train_percent=args.train_perc,
        valid_percent=args.valid_perc,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        plot_results=args.plot_results,
        batch_size=args.batch_size,
        logging_interval=args.logging_interval,
    )
