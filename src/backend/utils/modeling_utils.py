# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, sigmoid


def compute_metrics(
    labels: np.ndarray, predictions: np.ndarray, logits: Tensor
) -> Tuple[float, float, float, float, float]:
    """
    Compute various evaluation metrics for classification.

    Args:
        labels (np.ndarray): Ground truth binary labels.
        predictions (np.ndarray): Binary predictions from the model.
        logits (torch.Tensor): Raw model logits, before applying the sigmoid function.

    Returns:
        Tuple[float, float, float, float, float]:
        Tuple containing accuracy, precision, recall, F1 score, and ROC-AUC score.
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    aucroc = roc_auc_score(labels, sigmoid(logits).detach().cpu().numpy())
    return accuracy, precision, recall, f1, aucroc
