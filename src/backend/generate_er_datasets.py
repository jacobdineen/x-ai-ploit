# -*- coding: utf-8 -*-
"""Grab all documents from a directory of JSON files and write them to a CSV file.
Then Grab only documents that have no CVEs in either the cveids_db or cveids_explicit columns.
Used for entity recognition task
"""

import ast
import logging
import os

import pandas as pd
from tqdm import tqdm

# Initialize logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def parse_literal(column):
    """
    This function takes a string representation of a list and converts it to a list.

    Example:
    >>> parse_literal('[1, 2, 3]')
    [1, 2, 3]
    """
    try:
        return ast.literal_eval(column)
    except ValueError:
        return column


def get_tagged_and_untagged_documents(directory: str, output_csv: str):
    """
    This function takes a directory containing JSON files and writes them to a CSV file.
    Each JSON file is a single line in the CSV file.
    """
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    first_file = True

    for file in tqdm(json_files):
        file_path = os.path.join(directory, file)

        with open(file_path, "r") as f:
            for line in f:
                data = pd.read_json(line, lines=True)

                if first_file:
                    data.to_csv(output_csv, mode="w", index=False)
                    first_file = False
                else:
                    data.to_csv(output_csv, mode="a", index=False, header=False)


if __name__ == "__main__":
    directory = "/home/jdineen/Documents/xaiploit/data/copy20221006/documents/train_19700101_20210401"
    output_csv = "data/test.csv"
    output_cveless_csv = "data/cveless_docs.csv"
    output_cve_csv = "data/cve_docs.csv"
    regen_data = False

    if regen_data:
        get_tagged_and_untagged_documents(directory, output_csv)
        logging.info(f"all docs saved to {output_csv}")

        df = pd.read_csv(output_csv)
        logging.info(f"full df shape: {df.shape}")
        # Apply the conversion to the DataFrame
        df["cveids_db"] = df["cveids_db"].apply(parse_literal)
        df["cveids_explicit"] = df["cveids_explicit"].apply(parse_literal)

        # Now filter the DataFrame to get rows where both columns have empty lists
        filtered_df = df[
            (df["cveids_db"].map(lambda d: len(d) == 0))
            & (df["cveids_explicit"].map(lambda d: len(d) == 0))
        ]
        filtered_df.to_csv(output_cveless_csv, index=False)
        logging.info(f"cve-less df saved to {output_cveless_csv}")

        filtered_df = filtered_df.set_index(df.index[filtered_df.index])

        # Use the index to filter out the rows in 'filtered_df' from 'df'
        antiunion_df = df.drop(filtered_df.index)
        antiunion_df.to_csv(output_cve_csv, index=False)
        logging.info(f"cve df saved to {output_cve_csv}")

    cves = pd.read_csv(output_cve_csv)
    non_cves = pd.read_csv(output_cveless_csv)
    logging.info(
        f"""
    ------------------
    CVE Dataset Stats
    ------------------
    number of documents: {len(cves)}
    num unique cves: {len(cves['cveids_db'].unique())}
    avg docs per cve: {len(cves) / len(cves['cveids_db'].unique())}
    ------------------
    Unmapped-CVE Dataset Stats
    ------------------
    number of documents: {len(non_cves)}
    """
    )
