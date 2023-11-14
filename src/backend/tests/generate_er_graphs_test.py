# -*- coding: utf-8 -*-
import unittest

from src.backend import generate_er_graphs

# Rewrite the test below
# it fails with FAILED src/backend/tests/generate_er_graphs_test.py::TestGenerateErGraphs::test_create_inverse_mappings - AttributeError: 'NoneType' object has no attribute 'items'
# start now


class TestGenerateErGraphs(unittest.TestCase):
    def setUp(self):
        self.generator = generate_er_graphs.CVEGraphGenerator("data/cve_docs.csv")


if __name__ == "__main__":
    unittest.main()
