# -*- coding: utf-8 -*-
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from utils import *
from utils import timer

# Initialize logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def _get_do_labels(doc):
    source = doc["source"]

    if source == "xforce_api":
        return True
    elif source == "tenable_web":
        return False
    elif source == "tenable_plugins_web":
        return True
    elif source == "mitre_cvelist":
        return False
    elif source == "msrc_csvweb":
        return True
    elif source == "nvd_cves":
        return False
    elif source == "vulners_archive":
        doc_type = doc["content"]["_source"]["type"]
        if doc_type in ["nessus", "d2", "canvas", "metasploit"]:
            return True
        return False
    return False  # Default return


def _get_temporal_cvss_ecm(ecm):
    ecm = ecm.lower()
    if ecm == "functional":
        return "functional"
    if ecm == "high":
        return "high"
    if ecm in ["proof-of-concept", "proof of concept"]:
        return "proof-of-concept"
    if ecm == "unproven":
        return "unproven"
    if ecm in ["not defined", "not-defined", "none-found"]:
        return "not-defined"

    assert False, ">%s< entry not defined" % ecm


def _get_temporal_vector_exploitability_scores(tvs):
    min_score = 4

    for tv in tvs:
        if tv is None:
            continue
        expv = re.findall(r"E:(.*?)/", tv)[0]
        expvnum = None

        if expv == "F":
            expvnum = 0
        elif expv == "H":
            expvnum = 1
        elif expv == "POC":
            expvnum = 2
        elif expv == "P":
            expvnum = 2
        elif expv == "U":
            expvnum = 3
        elif expv == "ND":
            expvnum = 4
        elif expv == "X":
            expvnum = 4
        else:
            assert False, expv
        min_score = min(min_score, expvnum)
    return {
        0: "functional",
        1: "high",
        2: "proof-of-concept",
        3: "unproven",
        4: "not-defined",
    }[min_score]


def extract_labels(input_folder, output_folder):
    labels_fo = open(os.path.join(output_folder, "labels.json"), "w")

    for fl in tqdm(os.listdir(input_folder)):
        if ".json" not in fl:
            continue

        fi = open(os.path.join(input_folder, fl), "r")
        for row in fi:
            doc = json.loads(row)

            cveids = doc["cveids_explicit"]
            if len(cveids) != 1:
                continue
            cveid = cveids[0]

            publication_date = datetime.fromtimestamp(
                doc["date_published"]["$date"] / 1000.0
            ).isoformat()
            crawl_date = datetime.fromtimestamp(
                doc["date_crawl"]["$date"] / 1000.0
            ).isoformat()
            dict_base = {
                "cveid": cveid,
                "hash": doc["hash"],
                "crawl_date": crawl_date,
                "publication_date": publication_date,
            }
            do_labels = _get_do_labels(doc)
            label = None
            label_details = {}
            if do_labels:
                temporal_vectors_list = []
                if doc["source"] == "xforce_api":
                    # if doc['content'].get('exploitability',None) is None:
                    # 	pprint.pprint(doc)
                    # 	exit()
                    exploitability = doc["content"].get("exploitability", None)
                    if exploitability is None:
                        continue
                    label = _get_temporal_cvss_ecm(exploitability) in [
                        "functional",
                        "high",
                    ]
                    label_details["exploit_code_maturity"] = _get_temporal_cvss_ecm(
                        exploitability
                    )
                    label_details["exploit_url"] = (
                        "https://exchange.xforce.ibmcloud.com/vulnerabilities/%s"
                        % doc["content"]["xfdbid"]
                    )

                    # if cveid_exploitability_dict.get(cveid,None) is None:
                    # 	cveid_exploitability_dict[cveid] = set()
                    # cveid_exploitability_dict[cveid].add(exploitability)
                if doc["source"] == "vulners_archive":
                    src = doc["content"]["_source"]["type"]
                    if src == "nessus":
                        temporal_vectors_list = []
                        cvss_temporal2 = re.findall(
                            r"script_set_cvss_temporal_vector\(\s*\"(.+?)\"\s*\)",
                            "\n".join(doc["content"]["text"]),
                        )
                        assert len(cvss_temporal2) <= 1
                        if len(cvss_temporal2) == 1:
                            temporal_vectors_list.append(cvss_temporal2[0])

                        cvss_temporal3 = re.findall(
                            r"script_set_cvss3_temporal_vector\(\s*\"(.+?)\"\s*\)",
                            "\n".join(doc["content"]["text"]),
                        )
                        assert len(cvss_temporal3) <= 1
                        if len(cvss_temporal3) == 1:
                            temporal_vectors_list.append(cvss_temporal3[0])

                        # pprint.pprint(temporal_vectors_list)
                        # exit()
                        exploitability = _get_temporal_vector_exploitability_scores(
                            temporal_vectors_list
                        )
                        label = _get_temporal_cvss_ecm(exploitability) in [
                            "functional",
                            "high",
                        ]
                        label_details["exploit_code_maturity"] = exploitability
                        label_details["exploit_url"] = doc["content"]["_source"]["href"]
                    elif src == "d2":
                        label = True
                        label_details["exploit_evidence"] = "d2"
                        label_details["exploit_url"] = doc["content"]["_source"]["href"]
                    elif src == "canvas":
                        label = True
                        label_details["exploit_evidence"] = "canvas"
                        label_details["exploit_url"] = doc["content"]["_source"]["href"]
                        # pprint.pprint(doc)
                        # pprint.pprint(label_details)
                        # exit()
                    elif src == "metasploit":
                        label = True
                        label_details["exploit_evidence"] = "metasploit"
                        label_details["exploit_url"] = doc["content"]["_source"][
                            "sourceHref"
                        ]
                    else:
                        assert False, "Vulners source=%s not defined" % src
                if doc["source"] == "tenable_plugins_web":
                    temporal_vectors_list = []
                    cvss_temporal2 = (
                        doc["content"].get("cvss_v2_0", {}).get("temporal_vector", None)
                    )
                    if cvss_temporal2 is not None:
                        temporal_vectors_list.append(cvss_temporal2)

                    cvss_temporal3 = (
                        doc["content"].get("cvss_v3_0", {}).get("temporal_vector", None)
                    )
                    if cvss_temporal3 is not None:
                        temporal_vectors_list.append(cvss_temporal3)

                    if len(temporal_vectors_list) == 0:
                        continue
                    exploitability = _get_temporal_vector_exploitability_scores(
                        temporal_vectors_list
                    )
                    label = _get_temporal_cvss_ecm(exploitability) in [
                        "functional",
                        "high",
                    ]
                    label_details["exploit_code_maturity"] = exploitability
                    label_details[
                        "exploit_url"
                    ] = "https://www.tenable.com/plugins/nessus/%s" % (
                        doc["content"]["plugin_details"]["id"]
                    )

                if doc["source"] == "msrc_csvweb":
                    label = doc["content"]["Exploited"].lower() == "yes"
                    label_details["exploit_evidence"] = "msrc"
                    label_details["exploit_url"] = doc["content"]["CVE Number"]

            if label is not None:
                dict_label = dict(dict_base)
                dict_label["exploitability"] = label
                dict_label.update(label_details)
                labels_fo.write(json.dumps(dict_label) + "\n")

    logging.info("Done extracting labels.")

    labels_fo.close()


@timer
def extract_exploit_data(directory: str) -> List[Dict[str, Any]]:
    """
    Extracts specific fields from JSON files in the given directory.

    Parameters:
    - directory (str): Path to the directory containing the JSON files.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries containing the extracted data.
    """

    extracted_data = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure it's a file before reading
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                try:
                    file_content = json.load(f)

                    # Extract specific data using the given keys
                    strings = file_content.get("strings", [])
                    noncomments = file_content.get("noncomments", [])
                    comments = file_content.get("comments", [])

                    # Get filename without extension
                    filename_without_extension = os.path.splitext(filename)[0]

                    # Append the extracted data along with the filename without extension to the main list
                    extracted_data.append(
                        {
                            "hash": filename_without_extension,
                            "strings": strings,
                            "noncomments": noncomments,
                            "comments": comments,
                        }
                    )
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return extracted_data


@timer
def load_json_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    Loads a JSON file into a pandas DataFrame.

    Parameters:
    - filepath (str): Path to the JSON file.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the JSON file.
    """

    return pd.read_json(filepath, lines=True)


@timer
def merge_dataframes_on_column(
    df1: pd.DataFrame, df2: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Merges two DataFrames based on a specified column.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.
    - column_name (str): The name of the column to join on.

    Returns:
    - pd.DataFrame: Merged DataFrame if there are matching rows, otherwise an empty DataFrame.
    """

    merged_df = df1.merge(df2, on=column_name, how="inner")

    return merged_df if not merged_df.empty else pd.DataFrame()


def main(
    exploit_data_directory: str,
    documents_path: str,
    labels_directory: str,
    export_path: str,
    export_path_documents: str,
    export_path_labels: str,
):
    # Extracting exploit data and loading it into a DataFrame
    exploit_data = extract_exploit_data(exploit_data_directory)
    exploit_df = pd.DataFrame(exploit_data)
    logging.info("Exploit data loaded into DataFrame.")
    logging.info(f"Number of rows: {exploit_df.shape[0]}")

    documents_df = load_json_to_dataframe(documents_path)
    logging.info("Documents loaded into DataFrame.")
    logging.info(f"Number of rows: {documents_df.shape[0]}")

    # Merging the two DataFrames on "hash"
    # This is raw documents
    merged_df = merge_dataframes_on_column(documents_df, exploit_df, column_name="hash")
    logging.info("Merged DataFrame created.")
    logging.info(f"Number of rows: {merged_df.shape[0]}")
    merged_df.to_csv(f"{export_path_documents}.csv")
    logging.info(f"documents DataFrame saved to {export_path_documents}.")

    logging.info("extracting labels")
    extract_labels(input_folder=labels_directory, output_folder="data/")
    labels_df = pd.read_json("data/labels.json", lines=True)
    cols = [
        "hash",
        "exploitability",
        "exploit_code_maturity",
        "exploit_url",
        "exploit_evidence",
    ]
    labels_df = labels_df[cols]
    logging.info(f"Loaded {len(labels_df)} rows from the labels directory.")
    logging.info(labels_df.head())
    labels_df.to_csv(f"{export_path_labels}.csv")
    logging.info(f"labels DataFrame saved to {export_path_labels}.")

    # labels_df.to_csv("test.csv")
    final_merged_df = pd.merge(merged_df, labels_df, on="hash", how="left")
    final_merged_df["exploitability"] = final_merged_df["exploitability"].apply(
        lambda x: 1 if x else 0
    )
    logging.info("Final merged DataFrame created.")
    logging.info(f"Number of rows: {final_merged_df.shape[0]}")
    logging.info("Final merged DataFrame head:")
    logging.info(final_merged_df.head())

    # we get very few hash matches to anything other than vulner?
    print(final_merged_df["exploitability"].value_counts())
    final_merged_df.to_csv(f"{export_path}.csv")
    logging.info(f"Final merged DataFrame saved to {export_path}.")


if __name__ == "__main__":
    # figure out correct pathing later
    exploit_data_directory = (
        "data/copy20221006/files/train_19700101_20210401/exploits_text"
    )
    documents_path = "data/copy20221006/files/train_19700101_20210401/documents.json"
    labels_directory = "data/copy20221006/documents/train_19700101_20210401"
    export_path = "data/data"
    export_path_documents = "data/documents"
    export_path_labels = "data/labels"
    main(
        exploit_data_directory=exploit_data_directory,
        documents_path=documents_path,
        labels_directory=labels_directory,
        export_path=export_path,
        export_path_documents=export_path_documents,
        export_path_labels=export_path_labels,
    )

# This gives us the ground truth matches between the documents and the exploitability labels
# but it doesn't match everything
