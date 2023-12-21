"""
Generate a graph from CVE data.
Add edges between CVEs and hashes, vectorize
the text, and save the graph, node attributes, and vectorizer.
To be used for downstream tasks such as training a graph convolutional network (GCN).

"""
import argparse
import concurrent.futures
import logging
import os
import sys

import fasttext.util
import networkx as nx
import nltk
import numpy as np
import pandas as pd
from joblib import dump, load
from networkx.algorithms import bipartite
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def download_nltk_resources():
    """
    Downloads NLTK resources if they are not already available.

    This function checks if the required NLTK resources, such as stopwords and wordnet,
    are available in the NLTK data directories. If any of the resources are missing,
    they are downloaded using the nltk.download() function.

    Args:
        None

    Returns:
        None
    """
    # Define the NLTK resource directories
    nltk_data_path = nltk.data.path

    # Resources to check and download
    resources = {"corpora/stopwords": "stopwords", "corpora/wordnet": "wordnet"}

    for resource_path, resource_name in resources.items():
        # Check if each resource is available in any of the NLTK data directories
        if not any(os.path.exists(os.path.join(path, resource_path)) for path in nltk_data_path):
            print(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name)
        else:
            print(f"NLTK resource '{resource_name}' already downloaded.")


class CVEGraphGenerator:
    """
    Class to take an input dataframe and generate a graph from it.
    The graph is a networkx graph with CVEs as nodes and hashes as nodes.
    The edges are created between CVEs and hashes if the CVE and hash are connected.

    Args:
        file_path: The path to the CSV file containing CVE data.
        limit: The number of rows to read from the CSV file.
        fasttext_model: The path to the fasttext model to use for vectorization.

    Attributes:
        file_path: The path to the CSV file containing CVE data.
        limit: The number of rows to read from the CSV file.
        graph: The networkx graph.
        cveid_col: The name of the column containing CVE IDs.
        fasttext_model: The fasttext model to use for vectorization.

    """

    def __init__(self, file_path, fasttext_model: str = "cc.en.300.bin", limit=None):
        self.file_path = file_path
        self.limit = limit
        self.graph = nx.Graph()
        self.cveid_col = "cveids_explicit"
        fasttext.util.download_model("en", if_exists="ignore")  # 'en' for English
        self.ft_model = fasttext.load_model(fasttext_model)
        download_nltk_resources()

    def _is_graph_bipartite(self) -> bool:
        """
        Check if the graph is bipartite.

        Returns:
            bool: True if the graph is bipartite, False otherwise.
        """
        logging.info("Checking if graph is bipartite...")
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

        def vectorize_docs(docs):
            """
            Vectorize a batch of documents using FastText.

            Args:
                docs (iterable of str): An iterable of documents to vectorize.

            Returns:
                np.array: An array of vector representations of the documents.
            """

            # Pre-allocate an array for the vectors
            vectors = np.zeros((len(docs), self.ft_model.get_dimension()))

            # Process each document in the batch
            for i, doc in enumerate(docs):
                words = doc.split()
                if words:
                    word_vectors = [self.ft_model.get_word_vector(word) for word in words]
                    vectors[i] = np.mean(word_vectors, axis=0)

            return vectors

        def batch_lemmatize(docs):
            """
            Lemmatize a batch of documents.

            Args:
                docs (iterable of str): An iterable of documents to lemmatize.

            Returns:
                iterable of str: Lemmatized documents.
            """
            lemmatizer = WordNetLemmatizer()
            lemmatized_docs = [" ".join([lemmatizer.lemmatize(word) for word in doc.split()]) for doc in docs]
            return lemmatized_docs

        # Convert to lowercase and remove irrelevant characters
        texts = texts.str.lower().str.replace(r"[^a-zA-Z\s]", "", regex=True)
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        stop_words_pattern = r"\b" + r"\b|\b".join(stop_words) + r"\b"
        texts = texts.str.replace(stop_words_pattern, "", regex=True)
        # Lemmatization
        lemmatized_texts = batch_lemmatize(texts.values)
        # Vectorization
        # print(lemmatized_texts.values.shape)
        vectors = vectorize_docs(lemmatized_texts)
        logging.info(f"Vectorized {len(vectors)} documents.")
        logging.info(f"Vector shape: {vectors.shape}")
        return vectors

    def read_and_preprocess(self):
        """
        Reads and preprocesses the data from a CSV file.

        Returns:
            pd.DataFrame: The preprocessed data as a DataFrame.
        """

        def process_chunk(chunk):
            """
            Process a chunk of data by applying necessary transformations and returning the processed chunk.

            Args:
                chunk (pandas.DataFrame): The chunk of data to be processed.

            Returns:
                pandas.DataFrame: The processed chunk of data.

            Raises:
                Exception: If there is an error during the processing.

            """
            try:
                chunk[self.cveid_col] = chunk[self.cveid_col].apply(eval)
                chunk = chunk.explode(self.cveid_col)
                # Preprocess and vectorize text
                vectors = self.preprocess_text_series(chunk["content_text"])
                # Assign vectors to the DataFrame. Each vector becomes a list or a Series.
                chunk["embedding"] = vectors.tolist()
                return chunk

            except Exception as e:
                print(f"Error processing chunk: {e}")
                return pd.DataFrame()  # Return an empty DataFrame in case of error

        def chunk_generator(file_path, chunksize):
            """
            Generator function that reads a CSV file in chunks and yields each chunk.

            Args:
                file_path (str): The path to the CSV file.
                chunksize (int): The number of rows to read per chunk.

            Yields:
                pandas.DataFrame: A chunk of data read from the CSV file.

            Raises:
                StopIteration: Raised when there are no more chunks to read.

            """
            try:
                for chunk in pd.read_csv(file_path, chunksize=chunksize, nrows=self.limit):
                    yield chunk
            except StopIteration:
                return

        chunksize = 10000
        futures = []
        processed_chunks = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ix, chunk in enumerate(chunk_generator(self.file_path, chunksize)):
                logging.info(f"Processing chunk {ix}")
                future = executor.submit(process_chunk, chunk)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                processed_chunk = future.result()
                if not processed_chunk.empty:
                    processed_chunks.append(processed_chunk)

        df = pd.concat(processed_chunks, ignore_index=True) if processed_chunks else pd.DataFrame()
        return df

    def create_graph(self, df: pd.DataFrame) -> None:
        """
        Create a graph from the dataframe.

        Args:
            df: a pandas DataFrame containing the data from the CSV file
            vectors: numpy array of fasttext embeddings

        Returns:
            None
        """
        logging.info("Creating the graph with vectorized text...")

        cve_node_count = 0
        hash_node_count = 0
        multi_edge_node_count = 0
        skipped_cves = 0
        skipped_hash = 0

        for _, row in tqdm(df.iterrows()):
            cve_id, hash_, embedding = row["cveids_explicit"], row["hash"], row["embedding"]

            if pd.isna(cve_id):
                skipped_cves += 1
                continue

            if pd.isna(hash_):
                skipped_hash += 1
                continue

            if cve_id not in self.graph:
                self.graph.add_node(cve_id, vector=embedding)
                cve_node_count += 1

            if hash_ not in self.graph:
                self.graph.add_node(hash_, vector=embedding)
                hash_node_count += 1

            self.graph.add_edge(cve_id, hash_)

        # Count nodes with more than one edge
        for node in self.graph.nodes:
            if self.graph.degree(node) > 1:
                multi_edge_node_count += 1

        logging.info(
            f"Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.\n"
            f"Number of skipped CVEs: {skipped_cves}.\n"
            f"Number of skipped hashes: {skipped_hash}.\n"
            f"Number of CVE nodes: {cve_node_count}.\n"
            f"Number of hash nodes: {hash_node_count}.\n"
            f"Number of nodes with more than one edge: {multi_edge_node_count}"
        )

        is_bipartite = self._is_graph_bipartite()
        if is_bipartite:
            logging.info("Graph is bipartite.")
        else:
            logging.warning("Graph is not bipartite & will not be saved. Exiting...")
            sys.exit(1)

    def write_graph(self, graph_path, vectorizer_path) -> None:
        logging.info(f"Writing graph structure to {graph_path}...")
        dump(self.graph, graph_path)
        logging.info(f"Writing fasttext model to {vectorizer_path}...")
        self.ft_model.save_model(vectorizer_path)

    def load_graph(self, graph_path: str, vectorizer_path: str) -> None:
        logging.info(f"Loading graph structure from {graph_path}...")
        self.graph = load(graph_path)
        # Load vectorizer
        logging.info(f"Loading vectorizer from {vectorizer_path}...")
        self.ft_model = fasttext.load_model(vectorizer_path)


def main(read_path: str, output_dir: str, limit: int = None) -> None:
    """
    Run the graph generator.

    Args:
        read_path: The path to the CSV file containing CVE data.
        graph_save_path: The path to save the graph to.
        limit: The number of rows to read from the CSV file.

    Returns:
        None
    """
    # output_dir/samplesize_{limit}/graph.gml
    # output_dir/samplesize_{limit}/ft_model.bin
    # make the directory if it doesn't exist, else do nothing
    os.makedirs(os.path.join(output_dir, f"samplesize_{limit}"), exist_ok=True)
    graph_save_path = os.path.join(output_dir, f"samplesize_{limit}", "graph.gml")
    vectorizer_path = os.path.join(output_dir, f"samplesize_{limit}", "ft_model.bin")
    if not os.path.exists(graph_save_path) and not os.path.exists(vectorizer_path):
        os.makedirs(os.path.join(output_dir, f"samplesize_{limit}"), exist_ok=True)
        generator = CVEGraphGenerator(read_path, limit=limit)
        df = generator.read_and_preprocess()
        generator.create_graph(df)
        generator.write_graph(graph_save_path, vectorizer_path)
        generator.load_graph(graph_save_path, vectorizer_path)
        logging.info(f"Graph, features and vectorizer saved to {graph_save_path}")
    else:
        logging.info("paths exist.. skipping data gen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a graph from CVE data.")
    parser.add_argument(
        "--read_path", type=str, default="data/cve_docs.csv", help="Path to the CSV file containing CVE data."
    )
    parser.add_argument("--output_dir", type=str, default="data/", help="Path to the output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to read from the CSV file.")
    args = parser.parse_args()

    main(
        read_path=args.read_path,
        output_dir=args.output_dir,
        limit=args.limit,
    )
