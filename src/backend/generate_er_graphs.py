# -*- coding: utf-8 -*-
import argparse
import logging
import pickle
import re
import sys

import networkx as nx
import nltk
import pandas as pd
from networkx.algorithms import bipartite
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class CVEGraphGenerator:
    """
    Class to take an input dataframe and generate a graph from it.
    The graph is a networkx graph with CVEs as nodes and hashes as nodes.
    The edges are created between CVEs and hashes if the CVE and hash are connected.

    Args:
        file_path: The path to the CSV file containing CVE data.
        limit: The number of rows to read from the CSV file.

    Attributes:
        file_path: The path to the CSV file containing CVE data.
        limit: The number of rows to read from the CSV file.
        graph: The networkx graph.
        cveid_col: The name of the column containing CVE IDs.

    """

    def __init__(self, file_path, limit=None):
        self.file_path = file_path
        self.limit = limit
        self.graph = nx.Graph()
        self.cveid_col = "cveids_explicit"
        self.vectorizer = None
        nltk.download("stopwords")
        nltk.download("wordnet")

    def _is_graph_bipartite(self) -> bool:
        """
        Check if the graph is bipartite.

        Returns:
            bool: True if the graph is bipartite, False otherwise.
        """
        return bipartite.is_bipartite(self.graph)

    def preprocess_text(self, text):
        """
        Preprocess the given text by converting to lowercase, removing irrelevant characters,
        removing stopwords, and performing lemmatization.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text, re.I | re.A)
        tokens = text.split()
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def read_and_preprocess(self) -> pd.DataFrame:
        """
        read a csv file and preprocess it.

        Returns:
            df: a pandas DataFrame containing the data from the CSV file

        """
        logging.info("Reading and preprocessing the file...")
        df = pd.read_csv(self.file_path, nrows=self.limit)
        try:
            df[self.cveid_col] = df[self.cveid_col].apply(eval)
            df = df.explode(self.cveid_col)
        except Exception as e:
            logging.error(f"Error reading or processing the file: {e}")
        df["content_text"] = df["content_text"].apply(self.preprocess_text)
        return df

    def vectorize_text(self, df: pd.DataFrame):
        logging.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 1))
        vectors = vectorizer.fit_transform(df["content_text"])
        return vectors, vectorizer

    def create_graph(self, df: pd.DataFrame, vectors) -> None:
        """
        create a graph from the dataframe.

        Args:
            df: a pandas DataFrame containing the data from the CSV file

        Returns:
            None
        """
        logging.info("Creating the graph with vectorized text...")
        for index, row in tqdm(df.iterrows()):
            cve_id = row["cveids_explicit"]
            hash_ = row["hash"]

            if cve_id not in self.graph:
                self.graph.add_node(cve_id)

            if hash_ not in self.graph:
                vector = vectors[index].toarray()[0]  # Convert sparse matrix row to dense array
                self.graph.add_node(hash_, vector=vector)

            self.graph.add_edge(cve_id, hash_)
        logging.info(f"Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

        is_bipartite = self._is_graph_bipartite()
        if is_bipartite:
            logging.info("Graph is bipartite.")
        else:
            logging.warning("Graph is not bipartite.")
            logging.warning("Graph will not be saved.")
            logging.warning("Exiting...")
            sys.exit(1)

    def get_content_text(self, hash_) -> str:
        """
        Retrieve the content text for a given hash.

        Args:
            hash_: The hash identifier.

        Returns:
            The content text associated with the hash, if available.
        """
        return self.graph.nodes[hash_].get("content_text", None)

    def get_adjacency_matrix(self) -> nx.adjacency_matrix:
        """
        Create and return the adjacency matrix of the graph.

        Returns:
            A sparse matrix representing the adjacency matrix.
        """
        return nx.adjacency_matrix(self.graph)

    def write_graph(self, path, vectorizer) -> None:
        """
        Write the graph and the vectorizer to a file.

        Args:
            path: The path to write the graph and vectorizer to.
            vectorizer: The TfidfVectorizer object used for text vectorization.
        """
        with open(path, "wb") as f:
            pickle.dump((self.graph, vectorizer), f)

    def load_graph(self, path) -> None:
        """
        Load a graph and vectorizer from a file.

        Args:
            path: The path to load the graph and vectorizer from.
        """
        with open(path, "rb") as f:
            self.graph, self.vectorizer = pickle.load(f)


def main(read_path: str, graph_save_path: str, limit: int = None) -> None:
    """
    Run the graph generator.

    Args:
        read_path: The path to the CSV file containing CVE data.
        graph_save_path: The path to save the graph to.
        limit: The number of rows to read from the CSV file.

    Returns:
        None
    """
    generator = CVEGraphGenerator(read_path, limit=limit)
    df = generator.read_and_preprocess()
    vectors, vectorizer = generator.vectorize_text(df)
    generator.create_graph(df, vectors)
    generator.write_graph(graph_save_path, vectorizer)
    logging.info(f"Graph and vectorizer saved to {graph_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a graph from CVE data.")
    parser.add_argument(
        "--read_path", type=str, default="data/cve_docs.csv", help="Path to the CSV file containing CVE data."
    )
    parser.add_argument("--graph_save_path", type=str, default="data/graph.pkl", help="Path to the nx graph to")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to read from the CSV file.")
    args = parser.parse_args()

    main(read_path=args.read_path, graph_save_path=args.graph_save_path, limit=args.limit)


# # Initialize logging
# log_format = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=logging.INFO, format=log_format)


# class CVEGraphGenerator:
#     """
#     This class takes a CSV file containing CVEs and hashes and creates an adjacency matrix.
#     The adjacency matrix is a sparse matrix where the rows correspond to CVEs and the columns correspond to hashes.
#     The values in the matrix are 1 if the CVE and hash are connected and 0 otherwise.
#     """

#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.cveid_to_index = None
#         self.hash_to_index = None
#         self.index_to_cveid = None
#         self.index_to_hash = None
#         self.adj_matrix = None
#         self.cve_col = "cveids_explicit"
#         self.hash_to_content = None
#     def read_and_preprocess(self, nrows: int = None) -> pd.DataFrame:
#         """
#         This function reads the CSV file and preprocesses it.

#         Args:
#             nrows: the number of rows to read from the CSV file

#         Returns:
#             df: a pandas DataFrame containing the data from the CSV file
#         """
#         logging.info("Reading and preprocessing the file...")
#         df = pd.read_csv(self.file_path, nrows=nrows)
#         try:
#             df[self.cveid_col] = df[self.cveid_col].apply(eval)
#             df = df.explode(self.cveid_col)
#         except Exception as e:
#             logging.error(f"Error reading or processing the file: {e}")
#         self.hash_to_content = df.set_index('hash')['content_text'].to_dict()
#         return df

#     def create_mappings(self, df: pd.DataFrame) -> Tuple[dict, dict]:
#         """
#         Args:
#             df: a pandas DataFrame containing the data from the CSV file

#         Returns:
#             cveid_to_index: a dictionary mapping CVE IDs to indices
#             hash_to_index: a dictionary mapping hashes to indices
#         """
#         logging.info("Creating mappings...")
#         self.cveid_to_index = {cveid: idx for idx, cveid in enumerate(df["cveids_explicit"].unique())}
#         self.hash_to_index = {hash_: idx for idx, hash_ in enumerate(df["hash"].unique())}
#         return self.cveid_to_index, self.hash_to_index

#     def create_inverse_mappings(self) -> Tuple[dict, dict]:
#         """
#         Args:
#             cveid_to_index: a dictionary mapping CVE IDs to indices
#             hash_to_index: a dictionary mapping hashes to indices

#         Returns:
#             index_to_cveid: a dictionary mapping indices to CVE IDs
#             index_to_hash: a dictionary mapping indices to hashes
#         """
#         logging.info("Creating inverse mappings...")
#         self.index_to_cveid = {idx: cveid for cveid, idx in self.cveid_to_index.items()}
#         self.index_to_hash = {idx: hash_ for hash_, idx in self.hash_to_index.items()}
#         return self.index_to_cveid, self.index_to_hash

#     def create_adjacency_matrix(self, df: pd.DataFrame) -> coo_matrix:
#         """
#         Args:
#             df: a pandas DataFrame containing the data from the CSV file

#         Returns:
#             coo: a scipy sparse matrix containing the adjacency matrix
#         """
#         logging.info("Creating adjacency matrix...")
#         row_indices = df["cveids_explicit"].map(self.cveid_to_index).values
#         col_indices = df["hash"].map(self.hash_to_index).values
#         data = np.ones(len(df))
#         coo = coo_matrix(
#             (data, (row_indices, col_indices)),
#             shape=(len(self.cveid_to_index), len(self.hash_to_index)),
#         )
#         self.adj_matrix = coo
#         return self.adj_matrix

#     def find_non_zero_indices(self) -> Tuple[int, int]:
#         """
#         Args:
#             coo: a scipy sparse matrix containing the adjacency matrix

#         Returns:
#             non_zero_row: the row index of the first non-zero element in the adjacency matrix
#             non_zero_col: the column index of the first non-zero element in the adjacency matrix
#         """
#         non_zero_row, non_zero_col = np.nonzero(self.adj_matrix)[0][0], np.nonzero(self.adj_matrix)[1][0]
#         return non_zero_row, non_zero_col

#     def main(self, nrows: int = None) -> Tuple[coo_matrix, Tuple[dict, dict, dict, dict]]:
#         """
#         This function is the main function of the class. It calls all the other functions in the class.

#         Args:
#             nrows: the number of rows to read from the CSV file

#         Returns:
#             coo: a scipy sparse matrix containing the adjacency matrix
#             cveid_to_index: a dictionary mapping CVE IDs to indices
#             hash_to_index: a dictionary mapping hashes to indices
#             index_to_cveid: a dictionary mapping indices to CVE IDs
#             index_to_hash: a dictionary mapping indices to hashes
#         """
#         df = self.read_and_preprocess(nrows)
#         cveid_to_index, hash_to_index = self.create_mappings(df)
#         index_to_cveid, index_to_hash = self.create_inverse_mappings()
#         coo = self.create_adjacency_matrix(df)

#         logging.info(f"Adjacency matrix shape: {self.adj_matrix.shape}")
#         logging.info(f"# CVEs: {self.adj_matrix.shape[0]}")
#         logging.info(f"# Documents: {self.adj_matrix.shape[1]}")

#         non_zero_row, non_zero_col = self.find_non_zero_indices()
#         cveid = self.index_to_cveid[non_zero_row]
#         hash_ = self.index_to_hash[non_zero_col]

#         logging.info(f"Row index {non_zero_row} corresponds to CVE ID: {cveid}")
#         logging.info(f"Column index {non_zero_col} corresponds to Hash: {hash_}")
#         return coo, (cveid_to_index, hash_to_index, index_to_cveid, index_to_hash)

# # Function to convert string to dictionary and extract 'text'
# def extract_text(content):
#     try:
#         # Replace single quotes with double quotes
#         content = content.replace("'", '"')
#         # Convert string to dictionary
#         dict_content = json.loads(content)
#         # Extract 'text' if available
#         return dict_content['_source']['text'] if '_source' in dict_content and 'text' in dict_content['_source'] else None
#     except json.JSONDecodeError as e:
#         # Print the error for debugging
#         print("JSONDecodeError:", e, "in content:", content)
#         return None


# if __name__ == "__main__":
#     file_path = "data/cve_docs.csv"
#     generator = CVEGraphGenerator(file_path)
#     adjacency_matrix, mappings = generator.main()
