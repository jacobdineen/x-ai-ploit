# -*- coding: utf-8 -*-
"""
module to take in a csv file of CVE documents and generate an ER graph
in the form of a sparse adjacency matrix

Includes helper funcs to create mappings between CVE IDs and hashes
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


# pylint: disable=W0108
def read_and_preprocess(file_path, nrows=None):
    df = pd.read_csv(file_path, nrows=nrows)
    df["cveids_explicit"] = df["cveids_explicit"].apply(lambda x: eval(x))
    return df.explode("cveids_explicit")


def create_mappings(flat_df):
    cveid_to_index = {cveid: idx for idx, cveid in enumerate(flat_df["cveids_explicit"].unique())}
    hash_to_index = {hash_: idx for idx, hash_ in enumerate(flat_df["hash"].unique())}
    return cveid_to_index, hash_to_index


def create_inverse_mappings(cveid_to_index, hash_to_index):
    index_to_cveid = {idx: cveid for cveid, idx in cveid_to_index.items()}
    index_to_hash = {idx: hash_ for hash_, idx in hash_to_index.items()}
    return index_to_cveid, index_to_hash


def create_adjacency_matrix(flat_df, cveid_to_index, hash_to_index):
    row_indices = flat_df["cveids_explicit"].map(cveid_to_index).values
    col_indices = flat_df["hash"].map(hash_to_index).values
    data = np.ones(len(flat_df))
    coo = coo_matrix((data, (row_indices, col_indices)), shape=(len(cveid_to_index), len(hash_to_index)))
    return coo.toarray()


def find_non_zero_indices(matrix):
    non_zero_row, non_zero_col = np.nonzero(matrix)[0][0], np.nonzero(matrix)[1][0]
    return non_zero_row, non_zero_col


if __name__ == "__main__":
    # Main execution
    file_path = "data/cve_docs.csv"
    flat_df = read_and_preprocess(file_path, nrows=10)

    cveid_to_index, hash_to_index = create_mappings(flat_df)
    index_to_cveid, index_to_hash = create_inverse_mappings(cveid_to_index, hash_to_index)

    adj_matrix = create_adjacency_matrix(flat_df, cveid_to_index, hash_to_index)

    print(adj_matrix.shape)
    print(adj_matrix[:5])

    non_zero_row, non_zero_col = find_non_zero_indices(adj_matrix)
    cveid = index_to_cveid[non_zero_row]
    hash_ = index_to_hash[non_zero_col]

    print(f"Row index {non_zero_row} corresponds to CVE ID: {cveid}")
    print(f"Column index {non_zero_col} corresponds to Hash: {hash_}")
