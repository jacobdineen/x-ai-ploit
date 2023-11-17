# -*- coding: utf-8 -*-
"""
Grab all documents from a directory of JSON files and write them to a CSV file.
Then Grab only documents that have no CVEs in either the cveids_db or cveids_explicit columns.
Used for entity recognition task
"""
import argparse
import ast
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

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


def get_tagged_and_untagged_documents(directory: str, output_csv: str, limit: int = None):
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")][:limit]

    all_data = []

    for file in tqdm(json_files):
        file_path = os.path.join(directory, file)
        with open(file_path, "r") as f:
            for line in f:
                # Parse each line as JSON
                json_data = json.loads(line)

                # Extract required fields with nested structures
                data_row = {
                    "hash": json_data.get("hash", None),
                    "source": json_data.get("source", None),
                    "cveids_db": json_data.get("cveids_db", None),
                    "cveids_explicit": json_data.get("cveids_explicit", None),
                    "date_crawl": json_data.get("date_crawl", {}).get("$date", None),
                    "date_published": json_data.get("date_published", {}).get("$date", None),
                    "content_text": json_data.get("content", {}).get("text", None),
                }
                all_data.append(data_row)

    # Combine all data rows into a single DataFrame
    full_df = pd.DataFrame(all_data)

    # Write to CSV
    full_df.to_csv(output_csv, index=False)


def main(
    directory: str,
    output_csv: str,
    output_cve_csv: str,
    output_cveless_csv: str,
    regen_data: bool = False,
    limit: int = None,
):
    """
    Grab all documents from a directory of JSON files and write them to a CSV file.
    Then Grab only documents that have no CVEs in either the cveids_db or cveids_explicit columns.
    Used for entity recognition task

    Args:
        directory (str): Path to directory containing JSON files
        output_csv (str): Path to output CSV file
        output_cve_csv (str): Path to output CSV file containing only documents with CVEs
        output_cveless_csv (str): Path to output CSV file containing only documents without CVEs
        regen_data (bool, optional): Whether to regenerate the CSV files. Defaults to False.

    Returns:
        None
    """
    if regen_data:
        get_tagged_and_untagged_documents(directory, output_csv, limit=limit)
        logging.info(f"all docs saved to {output_csv}")

        df = pd.read_csv(output_csv)
        logging.info(f"full df shape: {df.shape}")
        # Apply the conversion to the DataFrame
        df["cveids_db"] = df["cveids_db"].apply(parse_literal)
        df["cveids_explicit"] = df["cveids_explicit"].apply(parse_literal)

        # Now filter the DataFrame to get rows where both columns have empty lists
        filtered_df = df[
            (df["cveids_db"].map(lambda d: len(d) == 0)) & (df["cveids_explicit"].map(lambda d: len(d) == 0))
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
    logging.info(cves.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files for entity recognition tasks.")
    parser.add_argument(
        "--directory",
        type=str,
        default="/home/jdineen/Documents/xaiploit/data/copy20221006/documents/train_19700101_20210401",
        help="Path to directory containing JSON files",
    )
    parser.add_argument("--output_csv", type=str, default="data/test.csv", help="Path to output CSV file")
    parser.add_argument(
        "--output_cve_csv",
        type=str,
        default="data/cve_docs.csv",
        help="Path to output CSV file containing only documents with CVEs",
    )
    parser.add_argument(
        "--output_cveless_csv",
        type=str,
        default="data/cveless_docs.csv",
        help="Path to output CSV file containing only documents without CVEs",
    )
    parser.add_argument(
        "--regen_data", type=bool, default=True, help="Whether to regenerate the CSV files. Defaults to True."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of JSON files to process. Defaults to None (no limit).",
    )

    args = parser.parse_args()

    main(
        directory=args.directory,
        output_csv=args.output_csv,
        output_cve_csv=args.output_cve_csv,
        output_cveless_csv=args.output_cveless_csv,
        regen_data=args.regen_data,
        limit=args.limit,
    )
