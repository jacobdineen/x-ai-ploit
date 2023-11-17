# -*- coding: utf-8 -*-
import unittest
from io import StringIO

import pandas as pd

from src.backend.generate_er_graphs import CVEGraphGenerator


class TestCVEGraphGenerator(unittest.TestCase):
    def setUp(self):
        # Sample CSV data
        self.csv_data = StringIO(
            """
cveids_explicit,hash,content_text
["CVE-2021-1234"],hash1,Sample text 1
["CVE-2021-1235"],hash2,Sample text 2
"""
        )
        self.file_path = "dummy_path.csv"
        self.limit = None
        self.graph_generator = CVEGraphGenerator(self.file_path, self.limit)
        self.df = pd.read_csv(self.csv_data)
        self.df["cveids_explicit"] = self.df["cveids_explicit"].apply(eval)
        self.df = self.df.explode("cveids_explicit")

    def test_read_and_preprocess(self):
        self.graph_generator.read_and_preprocess = lambda: self.df
        df_processed = self.graph_generator.read_and_preprocess()
        self.assertEqual(len(df_processed), len(self.df))

    def test_create_graph(self):
        self.graph_generator.create_graph(self.df)
        self.assertEqual(len(self.graph_generator.graph.nodes), 4)
        self.assertEqual(len(self.graph_generator.graph.edges), 2)

    def test_get_content_text(self):
        self.graph_generator.create_graph(self.df)
        content_text = self.graph_generator.get_content_text("hash1")
        self.assertEqual(content_text, "Sample text 1")


if __name__ == "__main__":
    unittest.main()
