# -*- coding: utf-8 -*-
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# Initialize logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class CVEGraphGenerator:
    """
    This class takes a CSV file containing CVEs and hashes and creates an adjacency matrix.
    The adjacency matrix is a sparse matrix where the rows correspond to CVEs and the columns correspond to hashes.
    The values in the matrix are 1 if the CVE and hash are connected and 0 otherwise.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.cveid_to_index = None
        self.hash_to_index = None
        self.index_to_cveid = None
        self.index_to_hash = None
        self.adj_matrix = None

    def read_and_preprocess(self, nrows: int = None) -> pd.DataFrame:
        """
        This function reads the CSV file and preprocesses it.

        Args:
            nrows: the number of rows to read from the CSV file

        Returns:
            df: a pandas DataFrame containing the data from the CSV file
        """
        logging.info("Reading and preprocessing the file...")
        try:
            df = pd.read_csv(self.file_path, nrows=nrows)
            df[self.cveid_col] = df[self.cveid_col].apply(eval)
            df = df.explode(self.cveid_col)
        except Exception as e:
            logging.error(f"Error reading or processing the file: {e}")
        return df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[dict, dict]:
        """
        Args:
            df: a pandas DataFrame containing the data from the CSV file

        Returns:
            cveid_to_index: a dictionary mapping CVE IDs to indices
            hash_to_index: a dictionary mapping hashes to indices
        """
        logging.info("Creating mappings...")
        self.cveid_to_index = {cveid: idx for idx, cveid in enumerate(df["cveids_explicit"].unique())}
        self.hash_to_index = {hash_: idx for idx, hash_ in enumerate(df["hash"].unique())}
        return self.cveid_to_index, self.hash_to_index

    def create_inverse_mappings(self) -> Tuple[dict, dict]:
        """
        Args:
            cveid_to_index: a dictionary mapping CVE IDs to indices
            hash_to_index: a dictionary mapping hashes to indices

        Returns:
            index_to_cveid: a dictionary mapping indices to CVE IDs
            index_to_hash: a dictionary mapping indices to hashes
        """
        logging.info("Creating inverse mappings...")
        self.index_to_cveid = {idx: cveid for cveid, idx in self.cveid_to_index.items()}
        self.index_to_hash = {idx: hash_ for hash_, idx in self.hash_to_index.items()}
        return self.index_to_cveid, self.index_to_hash

    def create_adjacency_matrix(self, df: pd.DataFrame) -> coo_matrix:
        """
        Args:
            df: a pandas DataFrame containing the data from the CSV file

        Returns:
            coo: a scipy sparse matrix containing the adjacency matrix
        """
        logging.info("Creating adjacency matrix...")
        row_indices = df["cveids_explicit"].map(self.cveid_to_index).values
        col_indices = df["hash"].map(self.hash_to_index).values
        data = np.ones(len(df))
        coo = coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.cveid_to_index), len(self.hash_to_index)),
        )
        self.adj_matrix = coo
        return self.adj_matrix

    def find_non_zero_indices(self) -> Tuple[int, int]:
        """
        Args:
            coo: a scipy sparse matrix containing the adjacency matrix

        Returns:
            non_zero_row: the row index of the first non-zero element in the adjacency matrix
            non_zero_col: the column index of the first non-zero element in the adjacency matrix
        """
        non_zero_row, non_zero_col = np.nonzero(self.adj_matrix)[0][0], np.nonzero(self.adj_matrix)[1][0]
        return non_zero_row, non_zero_col

    def main(self, nrows: int = None) -> Tuple[coo_matrix, Tuple[dict, dict, dict, dict]]:
        """
        This function is the main function of the class. It calls all the other functions in the class.

        Args:
            nrows: the number of rows to read from the CSV file

        Returns:
            coo: a scipy sparse matrix containing the adjacency matrix
            cveid_to_index: a dictionary mapping CVE IDs to indices
            hash_to_index: a dictionary mapping hashes to indices
            index_to_cveid: a dictionary mapping indices to CVE IDs
            index_to_hash: a dictionary mapping indices to hashes
        """
        df = self.read_and_preprocess(nrows)
        cveid_to_index, hash_to_index = self.create_mappings(df)
        index_to_cveid, index_to_hash = self.create_inverse_mappings()
        coo = self.create_adjacency_matrix(df)

        logging.info(f"Adjacency matrix shape: {self.adj_matrix.shape}")
        logging.info(f"# CVEs: {self.adj_matrix.shape[0]}")
        logging.info(f"# Documents: {self.adj_matrix.shape[1]}")

        non_zero_row, non_zero_col = self.find_non_zero_indices()
        cveid = self.index_to_cveid[non_zero_row]
        hash_ = self.index_to_hash[non_zero_col]

        logging.info(f"Row index {non_zero_row} corresponds to CVE ID: {cveid}")
        logging.info(f"Column index {non_zero_col} corresponds to Hash: {hash_}")
        return coo, (cveid_to_index, hash_to_index, index_to_cveid, index_to_hash)


if __name__ == "__main__":
    file_path = "data/cve_docs.csv"
    generator = CVEGraphGenerator(file_path)
    adjacency_matrix, mappings = generator.main()
