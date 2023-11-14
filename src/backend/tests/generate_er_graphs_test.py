# -*- coding: utf-8 -*-
import unittest
from io import StringIO

import pandas as pd

from src.backend.generate_er_graphs import CVEGraphGenerator


class TestCVEGraphGenerator(unittest.TestCase):
    def setUp(self):
        # Setup with dummy CSV data
        self.dummy_csv = StringIO(
            """cveids_explicit,hash
                                    ['CVE-1234', 'CVE-2345'],hash1
                                    ['CVE-3456'],hash2"""
        )

        self.generator = CVEGraphGenerator("dummy_path")

    def test_create_mappings(self):
        self.generator.read_and_preprocess = lambda: pd.DataFrame(
            {"cveids_explicit": ["CVE-1234", "CVE-2345", "CVE-3456"], "hash": ["hash1", "hash2", "hash3"]}
        )

        cveid_to_index, hash_to_index = self.generator.create_mappings(self.generator.read_and_preprocess())

        self.assertEqual(cveid_to_index, {"CVE-1234": 0, "CVE-2345": 1, "CVE-3456": 2})
        self.assertEqual(hash_to_index, {"hash1": 0, "hash2": 1, "hash3": 2})

    def test_create_inverse_mappings(self):
        self.generator.cveid_to_index = {"CVE-1234": 0, "CVE-2345": 1, "CVE-3456": 2}
        self.generator.hash_to_index = {"hash1": 0, "hash2": 1, "hash3": 2}

        index_to_cveid, index_to_hash = self.generator.create_inverse_mappings()

        self.assertEqual(index_to_cveid, {0: "CVE-1234", 1: "CVE-2345", 2: "CVE-3456"})
        self.assertEqual(index_to_hash, {0: "hash1", 1: "hash2", 2: "hash3"})

    def test_create_adjacency_matrix(self):
        df = pd.DataFrame(
            {"cveids_explicit": ["CVE-1234", "CVE-2345", "CVE-3456"], "hash": ["hash1", "hash2", "hash3"]}
        )

        self.generator.cveid_to_index = {"CVE-1234": 0, "CVE-2345": 1, "CVE-3456": 2}
        self.generator.hash_to_index = {"hash1": 0, "hash2": 1, "hash3": 2}

        adj_matrix = self.generator.create_adjacency_matrix(df)

        self.assertEqual(adj_matrix.count_nonzero(), 3)
        self.assertEqual(adj_matrix.shape, (3, 3))

    def test_main(self):
        self.generator.read_and_preprocess = lambda nrows: pd.DataFrame(
            {"cveids_explicit": ["CVE-1234", "CVE-2345", "CVE-3456"], "hash": ["hash1", "hash2", "hash3"]}
        )

        self.generator.create_mappings = lambda df: (
            {"CVE-1234": 0, "CVE-2345": 1, "CVE-3456": 2},
            {"hash1": 0, "hash2": 1, "hash3": 2},
        )

        self.generator.create_inverse_mappings = lambda: (
            {0: "CVE-1234", 1: "CVE-2345", 2: "CVE-3456"},
            {0: "hash1", 1: "hash2", 2: "hash3"},
        )

        # self.generator.create_adjacency_matrix = lambda df: coo_matrix(np.array([[0, 1], [1, 0]]))

        # adj_matrix, mappings = self.generator.main()

        # self.assertEqual(adj_matrix.count_nonzero(), 2)
        # self.assertEqual(len(mappings), 4)  # Ensure all mappings are returned


if __name__ == "__main__":
    unittest.main()
