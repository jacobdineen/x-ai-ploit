# -*- coding: utf-8 -*-
"""Grab all documents from a directory of JSON files and write them to a CSV file.
Then Grab only documents that have no CVEs in either the cveids_db or cveids_explicit columns.
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
        # Safely evaluate the string as a Python literal (list)
        return ast.literal_eval(column)
    except ValueError:
        # Return the value as-is if it's not a string representation of a list
        return column


def get_tagged_and_untagged_documents(directory: str, output_csv: str):
    """
    This function takes a directory containing JSON files and writes them to a CSV file.
    Each JSON file is a single line in the CSV file.
    """
    # Get a list of all json files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    # Write headers just once, assuming all JSON files have the same structure
    first_file = True

    for file in tqdm(json_files):
        file_path = os.path.join(directory, file)

        # Process each file line by line to handle memory efficiently
        with open(file_path, "r") as f:
            for line in f:
                # Each line is a complete JSON object
                data = pd.read_json(line, lines=True)

                # If first file, write header, else append without header
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
    run_full = False

    if run_full:
        get_tagged_and_untagged_documents(directory, output_csv)
        logging.info(f"all docs saved to {output_csv}")

    df = pd.read_csv(output_csv)
    logging.info(f"full df shape: {df.shape}")
    # Apply the conversion to the DataFrame
    df["cveids_db"] = df["cveids_db"].apply(parse_literal)
    df["cveids_explicit"] = df["cveids_explicit"].apply(parse_literal)

    # Now filter the DataFrame to get rows where both columns have empty lists
    filtered_df = df[(df["cveids_db"].map(lambda d: len(d) == 0)) & (df["cveids_explicit"].map(lambda d: len(d) == 0))]
    logging.info(f"cve-less df shape: {filtered_df.shape}")
    filtered_df.to_csv(output_cveless_csv, index=False)
    logging.info(f"cve-less df saved to {output_cveless_csv}")

    filtered_df = filtered_df.set_index(df.index[filtered_df.index])

    # Use the index to filter out the rows in 'filtered_df' from 'df'
    antiunion_df = df.drop(filtered_df.index)
    logging.info(f"cve df shape: {antiunion_df.shape}")
    logging.info(f"cve df saved to {output_cve_csv}")
