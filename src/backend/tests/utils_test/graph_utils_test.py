# # -*- coding: utf-8 -*-
# import unittest

# import torch
# from torch_geometric.data import Data

# from src.backend.utils.graph_utils import split_edges_and_sample_negatives


# class TestSplitEdgesAndSampleNegatives(unittest.TestCase):
#     def test_split_and_sample(self):
#         num_nodes = 10
#         edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
#         x = torch.randn((num_nodes, 3))
#         data = Data(x=x, edge_index=edge_index)

#         train_perc = 0.6
#         valid_perc = 0.2
#         train_data, val_data, test_data = split_edges_and_sample_negatives(data, train_perc, valid_perc)

#         # Assertions
#         total_edges = edge_index.size(1)
#         self.assertEqual(train_data.edge_index.size(1), int(total_edges * train_perc))
#         self.assertEqual(val_data.edge_index.size(1), int(total_edges * valid_perc))
#         self.assertEqual(
#             test_data.edge_index.size(1), total_edges - int(total_edges * train_perc) - int(total_edges * valid_perc)
#         )


# if __name__ == "__main__":
#     unittest.main()
