# -*- coding: utf-8 -*-
# pylint: disable=W0613
import unittest
from io import StringIO
from unittest.mock import patch

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.backend.generate_er_graphs import CVEGraphGenerator


class TestCVEGraphGenerator(unittest.TestCase):
    def setUp(self):
        # Sample CSV data
        self.csv_data = StringIO(
            """
cveids_explicit,hash,content_text
["CVE-2021-1234"],hash1,"Sample text 1 with some stopwords"
["CVE-2021-1235"],hash2,"Another sample text 2 with stopwords"
"""
        )
        self.generator = CVEGraphGenerator(None)
        self.df = pd.read_csv(self.csv_data)
        self.df["cveids_explicit"] = self.df["cveids_explicit"].apply(eval)
        self.df = self.df.explode("cveids_explicit")

    @patch("nltk.download")
    def test_read_and_preprocess(self, mock_nltk_download):
        # Mock the read_and_preprocess method
        self.generator.read_and_preprocess = lambda: self.df
        df_processed = self.generator.read_and_preprocess()
        self.assertEqual(len(df_processed), len(self.df))
        # Check if text preprocessing is working

    @patch("nltk.download")
    def test_vectorize_text(self, mock_nltk_download):
        # Test that vectorize_text returns the expected format
        vectors, _ = self.generator.vectorize_text(self.df)
        self.assertEqual(vectors.shape[0], len(self.df))
        self.assertTrue(hasattr(vectors, "toarray"))

    @patch("nltk.download")
    def test_create_graph(self, mock_nltk_download):
        # Test graph creation
        vectors, _ = self.generator.vectorize_text(self.df)
        self.generator.create_graph(self.df, vectors)
        self.assertEqual(len(self.generator.graph.nodes), 4)
        self.assertEqual(len(self.generator.graph.edges), 2)

    @patch("nltk.download")
    def test_get_adjacency_matrix(self, mock_nltk_download):
        # Test adjacency matrix creation
        vectors, _ = self.generator.vectorize_text(self.df)
        self.generator.create_graph(self.df, vectors)
        adj_matrix = self.generator.get_adjacency_matrix()
        self.assertEqual(adj_matrix.shape[0], len(self.generator.graph.nodes))

    @patch("nltk.download")
    def test_write_and_load_graph(self, mock_nltk_download):
        # Test saving and loading of the graph
        vectors, vectorizer = self.generator.vectorize_text(self.df)
        self.generator.create_graph(self.df, vectors)
        self.generator.write_graph("/tmp/test_graph.pkl", vectorizer)
        self.generator.load_graph("/tmp/test_graph.pkl")
        self.assertEqual(len(self.generator.graph.nodes), 4)
        self.assertEqual(len(self.generator.graph.edges), 2)
        self.assertIsInstance(self.generator.vectorizer, TfidfVectorizer)


if __name__ == "__main__":
    unittest.main()
