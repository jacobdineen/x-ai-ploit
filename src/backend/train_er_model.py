# -*- coding: utf-8 -*-
"""
Module to train a graph convolutional network (GCN) model on a given graph dataset.

"""
import logging
import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, negative_sampling
from tqdm import tqdm

from src.backend.gcn import GCN
from src.backend.utils.utils import load_graph

# inherit logging from entrypoint
logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
) -> Tuple[float, Tuple[float, float, float, float, float, str], torch.Tensor, np.ndarray, np.ndarray]:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The graph convolutional network (GCN) model to be trained.
        data (torch_geometric.data.Data): The data object containing graph data,
                including node features and edge indices.

    Returns:
        Tuple[float, Tuple[float, float, float, float, float, str], Tensor, np.ndarray, np.ndarray]:
        A tuple containing the loss for the epoch, a tuple
        of various evaluation metrics (accuracy, precision, recall, F1 score, ROC-AUC score, classification report),
        the model logits, binary predictions, and labels.
    """
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.vector, data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([data.edge_label, data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

    logits = model.decode(z, edge_label_index).view(-1)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, edge_label)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    predictions = torch.sigmoid(logits) > 0.5
    metrics = compute_metrics(edge_label, predictions, loss.item())
    torch.cuda.empty_cache()
    return loss.item(), metrics, logits.detach().cpu(), predictions.detach().cpu(), edge_label.detach().cpu()


@torch.no_grad()
def eval_epoch(model: torch.nn.Module, data: Data):
    model.eval()
    z = model.encode(data.vector, data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([data.edge_label, data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

    logits = model.decode(z, edge_label_index).view(-1)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, edge_label)
    torch.cuda.empty_cache()

    predictions = torch.sigmoid(logits) > 0.5
    metrics = compute_metrics(edge_label, predictions, loss.item())
    torch.cuda.empty_cache()
    return loss.item(), metrics, logits.detach().cpu(), predictions.detach().cpu(), edge_label.detach().cpu()


def prepare_data(data, validation_percent, test_percent) -> Tuple[Any, int]:
    transform = T.Compose(
        [
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(
                num_val=validation_percent, num_test=test_percent, is_undirected=True, add_negative_train_samples=False
            ),
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

    return train_data, val_data, test_data


def _plot_results(metrics):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train"]["loss"], label="Training Loss")
    plt.plot(metrics["val"]["loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("data/loss.png")
    plt.show()


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_loss", float("inf"))


def main(
    read_dir: str,
    data_size: int,
    num_layers: list,
    checkpoint_path: str = "models/checkpoint.pth.tar",
    load_from_checkpoint: bool = False,
    **kwargs,
):
    """
    main logic to grab data, train model, and plot results

    Args:
        read_dir (str): The directory containing the files to read.
        num_layers (list): The list of hidden dimensions for each layer.
        checkpoint_path (str): The path to save the model checkpoint.

    kwargs:
        train_percent (float): The percentage of data to use for training.
        valid_percent (float): The percentage of data to use for validation.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        dropout_rate (float): The dropout rate for the model.
        plot_results (bool): Whether to plot the training and validation loss over epochs.
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
    logging_interval = kwargs.get("logging_interval", 100)
    load_from_checkpoint = kwargs.get("load_from_checkpoint", False)

    # Seed here
    torch.manual_seed(42)
    num_features = 300  # hard coded for now
    model = GCN(num_features, 128, 64, num_layers=num_layers, dropout=dropout_rate).to(device)
    logging.info(f"model loaded onto device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    graph, _ = load_graph(graph_save_path, vectorizer_path)
    data = from_networkx(graph)
    train_data, val_data, _ = prepare_data(data, train_percent, valid_percent)

    metric_keys = ["loss", "accuracy", "precision", "recall", "f1"]
    metrics = {phase: {key: [] for key in metric_keys} for phase in ["train", "val"]}

    best_val_loss = float("inf")  # Initialize best validation loss for checkpointing
    for epoch in tqdm(range(num_epochs)):
        train_metrics = train_epoch(model, train_data, optimizer)
        val_metrics = eval_epoch(model, val_data)

        epoch_metrics = {
            "train": train_metrics,  # Directly use the tuple
            "val": val_metrics,  # Directly use the tuple
        }

        for phase in ["train", "val"]:
            for key, value in zip(metric_keys, epoch_metrics[phase]):
                metrics[phase][key].append(value)

        format_metrics = lambda metrics: " - ".join(
            f"{name}: {metric:.4f}" for name, metric in zip(metric_keys, metrics[1])
        )
        train_metrics_formatted = format_metrics(train_metrics)
        val_metrics_formatted = format_metrics(val_metrics)

        if epoch % logging_interval == 0:
            logging.info(f"Epoch {epoch+1}:")
            logging.info(f"Train Loss: {train_metrics[0]:.4f} - Val Loss: {val_metrics[0]:.4f}")
            logging.info(f"Train Metrics: {train_metrics_formatted}")
            logging.info(f"Val Metrics: {val_metrics_formatted}")

        # Checkpointing logic
        is_best = val_metrics[0] < best_val_loss
        if is_best:
            best_val_loss = val_metrics[0]
            logging.info("checkpointing model at epoch %d with best validation loss of %.3f", epoch + 1, best_val_loss)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                filename=checkpoint_path,
            )

    if plot_results:
        _plot_results(metrics)

    return None


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="GCN Training Script")
#     parser.add_argument("--read_dir", type=str, default="data", help="Path to the nx graph to")
#     parser.add_argument("--data_size", type=int, default=None, help="Number of samples to use for training")
#     parser.add_argument("--train_perc", type=float, default=0.70, help="Percent of data to use for training")
#     parser.add_argument("--valid_perc", type=float, default=0.20, help="Percent of data to use for validation")
#     parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
#     parser.add_argument("--num_layers", type=int, default=100, help="Number of GCN layers")
#     parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
#     parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
#     parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for the model")
#     parser.add_argument("--logging_interval", type=int, default=100, help="logging interval for metrics")
#     parser.add_argument(
#         "--checkpoint_path", type=str, default="models/checkpoint.pth.tar", help="GCN model checkpoint path"
#     )
#     parser.add_argument("--load_from_checkpoint", type=bool, default=False, help="load best model checkpoint")
#     parser.add_argument("--plot_results", type=bool, default=True, help="plot training and validation loss")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_arguments()
#     main(
#         args.read_dir,
#         args.data_size,
#         args.num_layers,
#         checkpoint_path=args.checkpoint_path,
#         load_from_checkpoint=args.load_from_checkpoint,
#         train_percent=args.train_perc,
#         valid_percent=args.valid_perc,
#         num_epochs=args.num_epochs,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         dropout_rate=args.dropout_rate,
#         plot_results=args.plot_results,
#         logging_interval=args.logging_interval,
#     )
