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
from src.backend.utils.graph_utils import split_edges_and_sample_negatives

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")


def compute_metrics(labels, predictions, loss):
    labels = torch.clamp(torch.round(labels), 0, 1).bool()
    preds = predictions.bool()
    # True Positives, False Positives, and False Negatives
    tp = torch.sum(preds & labels).item()
    fp = torch.sum(preds & ~labels).item()
    fn = torch.sum(~preds & labels).item()

    # Accuracy
    accuracy = torch.sum(preds == labels).item() / labels.numel()

    # Precision, Recall, and F1 Score
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
) -> Tuple[float, Tuple[float, float, float, float, float, str], torch.Tensor, np.ndarray, np.ndarray]:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The graph convolutional network (GCN) model to be trained.
        data (torch_geometric.data.Data): The data object containing graph data,
                including node features and edge indices.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        criterion (torch.nn.modules.loss._Loss): The loss function used for training.
        device (torch.device): The device (CPU or CUDA) on which the model is being trained.

    Returns:
        Tuple[float, Tuple[float, float, float, float, float, str], Tensor, np.ndarray, np.ndarray]:
        A tuple containing the loss for the epoch, a tuple
        of various evaluation metrics (accuracy, precision, recall, F1 score, ROC-AUC score, classification report),
        the model logits, binary predictions, and labels.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    z = model(data.x.to(device), data.edge_index.to(device))
    logits = model.decode(z, data.train_pos_edge_index.to(device), data.train_neg_edge_index.to(device))
    labels = torch.cat(
        [torch.ones(data.train_pos_edge_index.size(1)), torch.zeros(data.train_neg_edge_index.size(1))], dim=0
    ).to(device)

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    predictions = torch.sigmoid(logits) > 0.5

    metrics = compute_metrics(labels, predictions, loss)
    torch.cuda.empty_cache()
    return loss.item(), metrics, logits, predictions, labels


def eval_epoch(
    model: torch.nn.Module, data: Data, criterion: Any, device: torch.device
) -> Tuple[float, Tuple[float, float, float, float, float, str], torch.Tensor, np.ndarray, np.ndarray]:
    """
    Evaluate the model on validation or test data for one epoch.

    Args:
        model (Module): The graph convolutional network (GCN) model to be evaluated.
        data (Data): The data object from torch_geometric containing
            graph data including node features and edge indices for validation or testing.
        criterion (Any): The loss function used for evaluation.
        device (torch.device): The device (CPU or CUDA) on which the model is being evaluated.

    Returns:
        Tuple[float, Tuple[float, float, float, float, float, str], Tensor, np.ndarray, np.ndarray]:
        A tuple containing the loss for the epoch,
        a tuple of various evaluation metrics (accuracy, precision, recall,
        F1 score, ROC-AUC score, classification report),
        the model logits, binary predictions, and labels.
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

        metrics = compute_metrics(labels, predictions, loss)

    torch.cuda.empty_cache()
    return loss.item(), metrics, logits, predictions, labels


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
    logging_interval: int = 100,
    checkpoint_path: str = "models/checkpoint.pth.tar",
    load_from_checkpoint: bool = False,
    **kwargs,
):
    """
    main logic to grab data, train model, and plot results

    Args:
        read_dir (str): The directory containing the files to read.
        hidden_dims (list): The list of hidden dimensions for each layer.
        logging_interval (int): The interval at which to log metrics.
        checkpoint_path (str): The path to save the model checkpoint.
        load_from_checkpoint (bool): Whether to load the best model checkpoint.

    kwargs:
        train_percent (float): The percentage of data to use for training.
        valid_percent (float): The percentage of data to use for validation.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        dropout_rate (float): The dropout rate for the model.
        plot_results (bool): Whether to plot the training and validation loss over epochs.

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
    train_data, val_data, _ = split_edges_and_sample_negatives(data, train_perc=train_percent, valid_perc=valid_percent)

    metric_keys = ["loss", "accuracy", "precision", "recall", "f1"]
    metrics = {phase: {key: [] for key in metric_keys} for phase in ["train", "val"]}

    best_val_loss = float("inf")  # Initialize best validation loss for checkpointing
    for epoch in tqdm(range(num_epochs)):
        try:
            torch.cuda.synchronize(device)
            train_metrics = train_epoch(model, train_data, optimizer, criterion, device)
            val_metrics = eval_epoch(model, val_data, criterion, device)

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
                # validation confusion matrix
                # using sklearn
                # from sklearn.metrics import confusion_matrix

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.read_dir,
        args.data_size,
        args.hidden_dims,
        logging_interval=args.logging_interval,
        checkpoint_path=args.checkpoint_path,
        load_from_checkpoint=args.load_from_checkpoint,
        train_percent=args.train_perc,
        valid_percent=args.valid_perc,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        plot_results=args.plot_results,
    )
