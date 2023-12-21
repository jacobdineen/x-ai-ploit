# # -*- coding: utf-8 -*-
# import unittest

# import numpy as np
# import torch

# from src.backend.utils.modeling_utils import compute_metrics


# class TestComputeMetrics(unittest.TestCase):
#     def test_compute_metrics(self):
#         # Mock data
#         labels = np.array([1, 0, 1, 0])
#         predictions = np.array([1, 0, 0, 1])
#         logits = torch.tensor([2.0, -1.0, 0.5, -0.5])

#         expected_accuracy = 0.5
#         expected_precision = 0.5
#         expected_recall = 0.5
#         expected_f1 = 0.5

#         accuracy, precision, recall, f1, _ = compute_metrics(labels, predictions, logits)

#         # Assert each metric
#         self.assertAlmostEqual(accuracy, expected_accuracy, places=4)
#         self.assertAlmostEqual(precision, expected_precision, places=4)
#         self.assertAlmostEqual(recall, expected_recall, places=4)
#         self.assertAlmostEqual(f1, expected_f1, places=4)


# if __name__ == "__main__":
#     unittest.main()
