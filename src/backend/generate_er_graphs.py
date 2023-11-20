# -*- coding: utf-8 -*-
import argparse
import logging
import pickle
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

    def preprocess_text_series(self, texts):
        """
        Preprocess a Pandas Series of text by converting to lowercase, removing irrelevant characters,
        removing stopwords, and performing lemmatization.

        Args:
            texts (pd.Series): The Series of text to preprocess.

        Returns:
            pd.Series: The preprocessed text.
        """
        # Convert to lowercase and remove irrelevant characters
        texts = texts.str.lower()
        texts = texts.str.replace(r"[^a-zA-Z\s]", "", regex=True)

        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        def lemmatize_and_remove_stopwords(text):
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            return " ".join(tokens)

        return texts.apply(lemmatize_and_remove_stopwords)

    def read_and_preprocess(self) -> pd.DataFrame:
        """
        Read a CSV file in chunks and preprocess it.

        Returns:
            df: a pandas DataFrame containing the preprocessed data from the CSV file
        """
        logging.info("Reading and preprocessing the file...")
        chunksize = 10000  # Adjust this based on your memory constraints
        processed_chunks = []

        for chunk in pd.read_csv(self.file_path, chunksize=chunksize, nrows=self.limit):
            try:
                chunk[self.cveid_col] = chunk[self.cveid_col].apply(eval)
                chunk = chunk.explode(self.cveid_col)
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
                continue

            chunk["content_text"] = self.preprocess_text_series(chunk["content_text"])
            processed_chunks.append(chunk)

        logging.info("Concatenating processed chunks...")
        df = pd.concat(processed_chunks, ignore_index=True)
        return df

    def vectorize_text(self, df: pd.DataFrame):
        logging.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=5, ngram_range=(1, 1))
        vectors = vectorizer.fit_transform(df["content_text"])
        return vectors, vectorizer

    def create_graph(self, df: pd.DataFrame, vectors) -> None:
        """
        Create a graph from the dataframe.

        Args:
            df: a pandas DataFrame containing the data from the CSV file

        Returns:
            None
        """
        logging.info("Creating the graph with vectorized text...")

        cve_node_count = 0
        hash_node_count = 0
        multi_edge_node_count = 0

        for index, row in tqdm(df.iterrows()):
            cve_id = row["cveids_explicit"]
            hash_ = row["hash"]

            if cve_id not in self.graph:
                self.graph.add_node(cve_id)
                cve_node_count += 1

            if hash_ not in self.graph:
                vector = vectors[index].toarray()[0]  # Convert sparse matrix row to dense array
                self.graph.add_node(hash_, vector=vector)
                hash_node_count += 1

            self.graph.add_edge(cve_id, hash_)

        # Count nodes with more than one edge
        for node in self.graph.nodes:
            if self.graph.degree(node) > 1:
                multi_edge_node_count += 1

        logging.info(f"Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
        logging.info(f"Number of CVE nodes: {cve_node_count}")
        logging.info(f"Number of hash nodes: {hash_node_count}")
        logging.info(f"Number of nodes with more than one edge: {multi_edge_node_count}")

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
